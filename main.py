from modelscope import AutoModelForCausalLM, AutoTokenizer
import os

if __name__ == "__main__":
    # 设置自定义缓存目录
    os.environ['MODELSCOPE_CACHE'] = './custom_modelscope_cache'

    # 模型名称
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 输入文本
    input_text = "你好，世界！"

    # 将输入文本编码为模型输入
    inputs = tokenizer(input_text, return_tensors="pt")

    # 生成输出
    outputs = model.generate(**inputs)

    # 解码输出为文本
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(output_text)

    # 解码输出为文本
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(output_text)