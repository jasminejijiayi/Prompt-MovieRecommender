import json
import requests
import numpy as np
from code7 import construct_prompt, parse_output
import logging
import sys
import os

# 确保 result 文件夹存在
os.makedirs('result', exist_ok=True)

# 配置日志，保存到 result/test3.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('result', 'test7.3.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 重定向print到日志
class PrintLogger:
    def write(self, message):
        if message.strip():  # 只记录非空消息
            logging.info(message.strip())
    def flush(self):
        pass

sys.stdout = PrintLogger()

# OpenRouter API 设置
OPENROUTER_API_KEY = "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"  # 请替换为你的API密钥
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def call_llm_api(messages, model="deepseek/deepseek-chat-v3-0324:free", temperature=0):
    """
    调用OpenRouter API
    
    参数:
    messages (list): 消息列表，格式符合OpenAI API的格式
    model (str): 使用的模型名称
    temperature (float): 温度参数，控制输出的随机性
    
    返回:
    str: 模型的回复文本
    """
    # 打印API调用参数
    print("\n==== API调用参数 ====")
    print(f"模型: {model}")
    print(f"温度: {temperature}")
    
    # 打印发送的消息
    # print("\n==== 发送的消息 ====")
    #for i, msg in enumerate(messages):
        #print(f"消息 {i+1}:")
        #print(f"  角色: {msg['role']}")
        #print(f"  内容: {msg['content']}")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        print("\n==== 发送API请求 ====")
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()  # 如果请求失败，抛出异常
        
        result = response.json()
        print("\n==== API响应 ====")
        print(f"响应状态码: {response.status_code}")
        print(f"响应Headers: {dict(response.headers)}")
        print(f"完整响应JSON: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        content = result["choices"][0]["message"]["content"]
        print(f"\n模型回复内容:\n{content}")
        
        # 打印用量信息（如果API提供）
        if "usage" in result:
            usage = result["usage"]
            print("\n==== 用量信息 ====")
            print(f"输入tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"输出tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"总tokens: {usage.get('total_tokens', 'N/A')}")
        
        return content
    except Exception as e:
        print(f"\n==== API调用出错 ====")
        print(f"错误信息: {e}")
        if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, 'response'):
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return None

def calculate_ndcg_for_sample(predicted_list, ground_truth_item, k=10):
    """
    计算单个样本的NDCG@k
    
    参数:
    predicted_list (list): 预测的电影ID列表
    ground_truth_item (list): 真实的目标电影，格式为[movie_id, movie_name]
    k (int): 评估的推荐列表长度
    
    返回:
    float: NDCG@k的值
    """
    # 确保预测列表不超过k个元素
    predicted_list = predicted_list[:k]
    
    # 创建相关性列表，只有匹配ground_truth_item的电影ID时相关性为1，其他为0
    ground_truth_id = ground_truth_item[0]
    relevance = [1 if item == ground_truth_id else 0 for item in predicted_list]
    
    # 打印相关性列表详情
    print(f"相关性列表: {relevance}")
    
    # 如果目标电影不在预测列表中，NDCG为0
    if sum(relevance) == 0:
        print("目标电影不在预测列表中，NDCG=0")
        return 0
    
    # 计算DCG (Discounted Cumulative Gain)
    dcg = 0
    for i, rel in enumerate(relevance):
        if rel == 1:
            # DCG公式：rel_i / log2(i+2)，因为位置是从0开始的，所以要+2
            position = i + 1
            discount = np.log2(position + 1)
            contribution = rel / discount
            dcg += contribution
            print(f"位置{position}的DCG贡献: {rel}/log2({position+1}) = {contribution:.4f}")
    
    # 计算理想DCG (IDCG)
    # 理想情况下，相关的电影应该排在第一位
    idcg = 1.0  # 1/log2(1+1) = 1/log2(2) = 1
    
    # 计算NDCG
    ndcg = dcg / idcg
    print(f"DCG: {dcg:.4f}, IDCG: {idcg:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    
    return ndcg

def test_on_sample(sample_data):
    """
    在单个样本上测试推荐系统
    
    参数:
    sample_data (dict): 样本数据，包含用户历史和候选电影
    
    返回:
    float: NDCG@k的值
    """
    print("\n========== 测试开始 ==========\n")
    
    # 打印样本信息
    print("==== 样本信息 ====")
    print(f"用户ID: {sample_data['user_id']}")
    print(f"用户历史电影数量: {len(sample_data['item_list'])}")
    print(f"候选电影数量: {len(sample_data['candidates'])}")
    print(f"目标电影: {sample_data['target_item'][1]} (ID: {sample_data['target_item'][0]})")
    
    # 使用code3.py的函数构造提示词
    print("\n==== 构造提示词 ====")
    print("调用construct_prompt函数...")
    messages_template = construct_prompt(sample_data)
    print(f"生成了{len(messages_template)}条消息模板")
    
    # 执行第一轮对话：分析用户偏好
    print("\n==== 第一轮调用API(用户分析)结果 ====\n")
    # 第一轮只发送system和第一个user消息
    first_round_messages = [
        messages_template[0],  # system消息
        messages_template[1]   # 第一个user消息(分析用户偏好)
    ]
    print(f"发送了 {len(first_round_messages)} 条消息进行用户分析（system + user第一轮消息）")
    
    # 调用API获取第一轮回复
    first_response = call_llm_api(first_round_messages)
    if not first_response:
        print("\n==== 错误 ====\n")
        print("第一轮API调用失败，无法获取响应")
        return 0
    
    # 执行第二轮对话：分析候选电影
    print("\n==== 第二轮调用API(候选电影分析)结果 ====\n")
    # 第二轮发送前两轮对话历史和第二个user消息
    second_round_messages = [
        messages_template[0],  # system消息
        {"role": "assistant", "content": first_response},  # 第一轮助手回复
        messages_template[3]   # 第二个user消息(分析候选电影)
    ]
    print(f"发送了 {len(second_round_messages)} 条消息进行候选电影分析（包含前两轮历史）")
    
    # 调用API获取第二轮回复
    second_response = call_llm_api(second_round_messages)
    if not second_response:
        print("\n==== 错误 ====\n")
        print("第二轮API调用失败，无法获取响应")
        return 0
    
    # 执行第三轮对话：要求输出最终推荐
    print("\n==== 第三轮调用API(最终推荐)结果 ====")
    # 第三轮发送完整对话历史和第三个user消息
    final_round_messages = [
        messages_template[0],  # system消息
        {"role": "assistant", "content": first_response},  # 第一轮助手回复
        {"role": "assistant", "content": second_response},  # 第二轮助手回复
        messages_template[5]   # 第三个user消息(要求输出JSON)
    ]
    print("==== 发送的消息 (最终推荐请求) ====")
    print(f"[用户第三轮 - 输出要求]:\n  {final_round_messages[3]['content']}")
    
    # 调用API获取最终回复
    response_text = call_llm_api(final_round_messages)
    if not response_text:
        print("\n==== 错误 ====")
        print("最终轮API调用失败，无法获取响应")
        return 0
    
    # 解析输出，获取推荐列表
    print("\n==== 解析输出 ====")
    print(f"调用parse_output函数解析响应...")
    predicted_list = parse_output(response_text)
    print(f"解析结果: {predicted_list}")
    
    # 计算NDCG@10
    print("\n==== 计算NDCG ====")
    target_item = sample_data["target_item"]
    target_id = target_item[0]
    k = 10
    ndcg = calculate_ndcg_for_sample(predicted_list, target_item, k=k)
    
    # 打印详细的NDCG计算过程
    print("NDCG@10计算详情:")
    predicted_k = predicted_list[:k]
    print(f"截断后的预测列表(前{k}个): {predicted_k}")
    
    # 显示目标电影在预测列表中的位置
    if target_id in predicted_k:
        position = predicted_k.index(target_id) + 1
        print(f"目标电影ID {target_id}在位置{position}")
        print(f"位置{position}的DCG贡献: 1/log2({position+1}) = {1/np.log2(position+1):.4f}")
    else:
        print(f"目标电影ID {target_id}不在前{k}个预测列表中")
    
    # 汇总结果
    print("\n==== 测试结果 ====")
    print(f"用户ID: {sample_data['user_id']}")
    print(f"目标电影: {target_item[1]} (ID: {target_id})")
    print(f"预测列表(前{k}个): {predicted_k}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    
    print("\n========== 测试结束 ==========\n")
    return ndcg

def main():
    """
    主函数，从val.jsonl加载数据并进行测试
    """
    print("开始执行测试...")
    
    # 读取验证数据集
    try:
        print("读取val.jsonl文件...")
        # 读取所有样本数据
        with open("val.jsonl", "r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f]
        print(f"成功加载{len(samples)}个样本")
    except Exception as e:
        print(f"读取数据出错: {e}")
        return
    
    # 选择前10个样本进行测试
    test_samples = samples[:10] if len(samples) >= 10 else samples
    num_samples = len(test_samples)
    print(f"将测试{num_samples}个样本")
    
    # 存储每个样本的NDCG@10值
    ndcg_values = []
    
    # 依次测试每个样本
    for i, sample in enumerate(test_samples):
        print(f"\n===== 测试样本 {i+1}/{num_samples} =====\n")
        # 测试单个样本
        ndcg = test_on_sample(sample)
        ndcg_values.append(ndcg)
        print(f"样本 {i+1} 测试完成，NDCG@10: {ndcg:.4f}")
    
    # 计算平均NDCG@10
    avg_ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0
    
    # 输出所有样本的NDCG@10值和平均值
    print("\n===== 测试总结 =====\n")
    for i, ndcg in enumerate(ndcg_values):
        print(f"样本 {i+1} NDCG@10: {ndcg:.4f}")
    print(f"\n总计测试了 {len(ndcg_values)} 个样本")
    print(f"平均NDCG@10: {avg_ndcg:.4f}")
    
    # 打印运行信息
    print("\n==== 运行信息 ====")
    print(f"执行时间: {np.datetime64('now')}")
    print(f"API密钥: {OPENROUTER_API_KEY[:8]}...{OPENROUTER_API_KEY[-8:]}")
    print(f"API地址: {API_URL}")

if __name__ == "__main__":
    main()
