from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Union
import os
from flask import Flask, request, jsonify
import time
import logging
from queue import Queue
import threading
import torch

# 设置自定义缓存目录
os.environ['MODELSCOPE_CACHE'] = './custom_modelscope_cache'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformers_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局模型和tokenizer实例
global_model = None
global_tokenizer = None
request_queue = Queue()
processing_lock = threading.Lock()

def initialize_model():
    global global_model, global_tokenizer
    if global_model is None:
        logger.info("正在初始化全局模型实例")
        # 使用本地已下载的模型文件
        model_dir = './custom_modelscope_cache/hub/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
        global_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        global_tokenizer = AutoTokenizer.from_pretrained(
            model_dir
        )
        logger.info("全局模型实例初始化完成")

class TransformersInference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        logger.info(f"正在生成响应，输入长度: {len(prompt)}")
        start_time = time.time()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )
        
        duration = time.time() - start_time
        logger.info(f"Transformers生成完成，耗时: {duration:.2f}秒")
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def format_chat_messages(messages):
    """将OpenAI格式的消息转换为模型输入"""
    formatted_text = ""
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        if role == 'system':
            formatted_text += f"System: {content}\n"
        elif role == 'user':
            formatted_text += f"Human: {content}\n"
        elif role == 'assistant':
            formatted_text += f"Assistant: {content}\n"
    return formatted_text.strip()

# 在应用启动时初始化模型
initialize_model()

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        request_id = f"req-{int(time.time()*1000)}"
        logger.info(f"收到请求 - IP: {request.remote_addr} - ID: {request_id}")
        logger.info(f"请求内容: {data}")
        
        # 验证消息结构
        if not isinstance(data.get('messages'), list):
            error_msg = 'messages must be a list'
            logger.error(f"请求错误: {error_msg} - ID: {request_id}")
            return jsonify({'error': error_msg}), 400

        messages = data.get('messages', [])

        if not messages:
            error_msg = 'messages field is required'
            logger.error(f"请求错误: {error_msg} - ID: {request_id}")
            return jsonify({'error': error_msg}), 400
            
        # 格式化消息
        input_text = format_chat_messages(messages)

        # 创建推理实例
        inference = TransformersInference(global_model, global_tokenizer)

        # 记录开始时间
        start_time = time.time()

        # 生成回复
        output_text = inference.generate(input_text)

        # 提取模型的回复
        model_reply = output_text.split("Assistant:")[-1].strip()

        # 计算结束时间
        end_time = time.time()

        response = {
            'id': request_id,
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'deepseek-r1-distill-qwen-14b',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': model_reply
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(input_text),
                'completion_tokens': len(model_reply),
                'total_tokens': len(input_text) + len(model_reply)
            }
        }

        logger.info(f"响应内容: {response}")
        logger.info(f"处理时间: {end_time - start_time:.2f}秒")

        return jsonify(response)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"处理请求时发生错误: {error_msg}")
        return jsonify({'error': error_msg}), 500

if __name__ == "__main__":
    # 确保模型加载完成后再启动服务
    logger.info("正在启动Flask应用...")
    app.run(host='0.0.0.0', port=5001, threaded=True)
