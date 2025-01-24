from vllm import LLM, SamplingParams
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from typing import Union
import os
from flask import Flask, request, jsonify
import time
import logging
from queue import Queue
import threading

# 设置自定义缓存目录
os.environ['MODELSCOPE_CACHE'] = './custom_modelscope_cache'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vllm_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局LLM实例
global_llm = None
request_queue = Queue()
processing_lock = threading.Lock()

def initialize_llm():
    global global_llm
    if global_llm is None:
        logger.info("正在初始化全局LLM实例")
        model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')
        global_llm = LLM(
            model=model_dir,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,  # 使用1个GPU
            max_num_batched_tokens=16384,  # 增加批处理token数量
            max_num_seqs=32,  # 适当降低最大序列数
            enable_prefix_caching=True,  # 启用前缀缓存
            enforce_eager=True  # 启用eager模式
        )
        logger.info("全局LLM实例初始化完成，优化配置已启用")

class VLLMInference:
    def __init__(self, llm_instance: LLM, sampling_params: SamplingParams):
        self.llm = llm_instance
        self.sampling_params = sampling_params

    def generate(self, prompt: str) -> str:
        logger.info(f"正在生成响应，输入长度: {len(prompt)}")
        start_time = time.time()
        
        with processing_lock:
            outputs = self.llm.generate([prompt], self.sampling_params)
        
        duration = time.time() - start_time
        logger.info(f"vLLM生成完成，耗时: {duration:.2f}秒")
        return outputs[0].outputs[0].text

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
initialize_llm()

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        request_id = f"req-{int(time.time()*1000)}"
        logger.info(f"收到请求 - IP: {request.remote_addr} - ID: {request_id}")
        logger.info(f"请求内容: {data}")
        
        # Validate messages structure
        if not isinstance(data.get('messages'), list):
            error_msg = 'messages must be a list'
            logger.error(f"请求错误: {error_msg} - ID: {request_id}")
            return jsonify({'error': error_msg}), 400

        messages = data.get('messages', [])

        if not messages:
            error_msg = 'messages field is required'
            logger.error(f"请求错误: {error_msg} - ID: {request_id}")
            return jsonify({'error': error_msg}), 400
            
        # Clear KV cache between requests
        try:
            global_llm.reset()
            logger.info(f"KV cache cleared - ID: {request_id}")
        except Exception as e:
            logger.warning(f"Failed to clear KV cache: {str(e)} - ID: {request_id}")

        # 格式化消息
        input_text = format_chat_messages(messages)

        # 创建推理实例
        inference = VLLMInference(
            global_llm,
            SamplingParams(
                temperature=0.8,  # 提高temperature增加多样性
                top_p=0.95,  # 提高top_p增加生成质量
                max_tokens=2048,  # 降低最大生成token数
                presence_penalty=0.1,  # 添加存在惩罚
                frequency_penalty=0.1,  # 添加频率惩罚
                skip_special_tokens=True  # 跳过特殊token
            )
        )

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
