from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
from flask import Flask, request, jsonify
import time
import torch
from flask.json.provider import JSONProvider
import json
import logging

# 设置自定义缓存目录
os.environ['MODELSCOPE_CACHE'] = './custom_modelscope_cache'

# 在创建Flask应用后添加自定义JSON提供程序
class CustomJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        # 确保中文不会被转换为 unicode 编码
        return json.dumps(obj, ensure_ascii=False, **kwargs)

    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)

app = Flask(__name__)
app.json = CustomJSONProvider(app)

# 全局变量存储模型和分词器
tokenizer = None
model = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model():
    global tokenizer, model
    # 检查GPU是否可用
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU:")
        total_memory = 0
        for i in range(gpu_count):
            gpu_properties = torch.cuda.get_device_properties(i)
            memory_gb = gpu_properties.total_memory / 1024**3
            total_memory += memory_gb
            print(f"GPU {i}: {gpu_properties.name}, 显存: {memory_gb:.2f} GB")
        print(f"总显存: {total_memory:.2f} GB")
    else:
        print("警告: 未检测到GPU，模型将在CPU上运行，这可能会很慢！")

    # 模型名称
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

    print("开始加载模型...")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # 自动在可用GPU之间分配模型层
        torch_dtype=torch.float16,  # 使用float16以减少显存占用
    )
    print(f"模型加载完成，使用设备映射: {model.hf_device_map if hasattr(model, 'hf_device_map') else model.device}")
    model.eval()  # 设置为评估模式


@app.route('/')
def hello():
    return "Hello, World!"


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


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        # 记录请求信息
        logger.info(f"收到请求 - IP: {request.remote_addr}")
        logger.info(f"请求内容: {data}")
        
        messages = data.get('messages', [])
        
        if not messages:
            error_msg = 'messages field is required'
            logger.error(f"请求错误: {error_msg}")
            return jsonify({'error': error_msg}), 400

        # 格式化消息
        input_text = format_chat_messages(messages)
        
        # 记录开始时间
        start_time = time.time()

        # 将输入文本编码为模型输入，确保在正确的设备上
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # 使用no_grad进行推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 1.0),
            )

        # 解码输出为文本
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"原来回复为：{output_text}")
        # 提取模型的回复（最后一个Assistant的回复）
        model_reply = output_text.split("Assistant:")[-1].strip()
        # 计算结束时间
        end_time = time.time()

        response = {
            'id': f'chatcmpl-{int(time.time())}',
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
                'prompt_tokens': len(inputs.input_ids[0]),
                'completion_tokens': len(outputs[0]) - len(inputs.input_ids[0]),
                'total_tokens': len(outputs[0])
            }
        }

        # 记录响应信息
        logger.info(f"响应内容1: {response}")
        logger.info(f"响应内容2: {json.dumps(response, ensure_ascii=False)}")
        logger.info(f"处理时间: {end_time - start_time:.2f}秒")
        
        return jsonify(response)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"处理请求时发生错误: {error_msg}")
        return jsonify({'error': error_msg}), 500


if __name__ == "__main__":
    # 启动时加载模型
    load_model()
    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000)
