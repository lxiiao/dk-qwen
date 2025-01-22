from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
from flask import Flask, request, jsonify
import time
import torch

# 设置自定义缓存目录
os.environ['MODELSCOPE_CACHE'] = './custom_modelscope_cache'

app = Flask(__name__)

# 全局变量存储模型和分词器
tokenizer = None
model = None


def load_model():
    global tokenizer, model
    # 模型名称
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print("开始加载模型...")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # 自动选择最优设备
        torch_dtype=torch.float16  # 使用float16以减少显存占用
    )
    print(f"模型加载完成，使用设备: {model.device}")
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
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': 'messages field is required'}), 400

        # 格式化消息
        input_text = format_chat_messages(messages)
        
        # 记录开始时间
        start_time = time.time()

        # 将输入文本编码为模型输入
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # 使用no_grad进行推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=data.get('temperature', 0.7),
                top_p=data.get('top_p', 1.0),
            )

        # 解码输出为文本
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取模型的回复（最后一个Assistant的回复）
        model_reply = output_text.split("Assistant:")[-1].strip()
        
        # 计算结束时间
        end_time = time.time()

        response = {
            'id': f'chatcmpl-{int(time.time())}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': 'deepseek-r1-distill-qwen-1.5b',
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

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # 启动时加载模型
    load_model()
    # 启动Flask服务
    app.run(host='0.0.0.0', port=5000)
