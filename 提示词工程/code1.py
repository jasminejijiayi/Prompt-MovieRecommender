import json
import re

def construct_prompt(d):
    """
    构造用于大语言模型的提示词
    
    参数:
    d (dict): jsonl数据文件的一行，为字典类型的变量，详细的数据格式见val.jsonl的说明
    
    返回:
    list: OpenAI API的message格式列表，允许设计多轮对话式的prompt
    示例: [{"role": "system", "content": "系统提示内容"}, 
           {"role": "user", "content": "用户提示内容"}]
    """
    # 提取用户ID
    user_id = d["user_id"]
    
    # 提取用户观看过的电影列表
    watched_movies = d["item_list"]
    watched_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in watched_movies])
    
    # 提取候选电影列表
    candidate_movies = d["candidates"]
    candidate_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in candidate_movies])
    
    # 构造系统提示
    system_content = (
        "你是一个专业的电影推荐系统。"    
        "根据用户的观影历史，你需要将候选电影列表重新排序，使得最可能被用户喜欢的电影排在前面。"    
        "请考虑电影的题材、风格、主题、导演、演员等多个因素，找出最适合用户的电影。"
        "为了确保推荐有效，请只从候选电影列表中进行推荐，不要添加任何列表以外的电影。"
        "输出格式要求：按照推荐优先级输出电影名称和ID的列表，使用JSON数组格式，每个元素包含电影名称和ID，不要包含其他说明文字，例如:[{\"name\": \"Bell, Book and Candle\", \"id\": 3516}, {\"name\": \"Max Dugan Returns\", \"id\": 3497}, ...]"   
    )
    
    # 构造用户提示
    user_content = (
        f"用户ID: {user_id}\n\n"
        f"用户观看过的电影列表:\n{watched_movies_text}\n\n"
        f"候选电影列表，请根据用户可能的喜好对以下电影名称进行排序:\n{candidate_movies_text}\n\n"
        "请直接输出按推荐优先级排序的电影名称和ID列表，使用JSON数组格式，每个元素包含电影名称和ID，不要包含其他解释。"
    )
    
    # 返回OpenAI API格式的提示词
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def parse_output(text):
    """
    解析大语言模型的输出文本，提取推荐重排列表
    
    参数:
    text (str): 大语言模型在设计prompt下的输出文本，按推荐优先级排序的电影名称和ID列表，使用JSON数组格式
    
    返回:
    list: 从输出文本解析出的电影ID列表（python列表格式，列表的每个元素是整数，表示编号），表示重排后的推荐顺序
    示例: [1893, 3148, 111, ...]
    """
    # 清理文本
    text = text.strip()
    
    # 尝试解析JSON数组格式的电影数据
    try:
        # 直接尝试解析整个文本
        movies_data = json.loads(text)
        if isinstance(movies_data, list):
            # 如果是电影名称和ID的字典列表格式
            if movies_data and isinstance(movies_data[0], dict) and 'id' in movies_data[0]:
                movie_ids = [movie['id'] for movie in movies_data if 'id' in movie]
                return movie_ids
            # 如果只是电影ID的列表
            elif movies_data and isinstance(movies_data[0], int):
                return movies_data
            # 如果只是电影名称的列表，需要通过候选电影名单查找对应ID
            elif movies_data and isinstance(movies_data[0], str):
                # 尝试读取测试样本中的候选电影信息
                try:
                    with open("val.jsonl", "r", encoding="utf-8") as f:
                        # 读取所有行，确保我们有正在测试的样本
                        lines = f.readlines()
                        # 尝试处理所有可用的样本数据
                        all_candidates = []
                        for line in lines:
                            sample = json.loads(line)
                            candidates = sample.get("candidates", [])
                            all_candidates.extend(candidates)
                        
                        # 创建电影名称到ID的映射
                        name_to_id = {}
                        for movie_id, movie_name in all_candidates:
                            name_to_id[movie_name] = movie_id
                        
                        # 将电影名称映射到ID
                        movie_ids = []
                        for name in movies_data:
                            if name in name_to_id:
                                movie_ids.append(name_to_id[name])
                        return movie_ids
                except Exception:
                    # 如果无法读取样本文件，尝试其他方法
                    pass
    except json.JSONDecodeError:
        # 如果解析失败，尝试找到文本中的JSON数组部分
        pattern = r'\[(.*?)\]'
        matches = re.search(pattern, text)
        if matches:
            try:
                # 尝试解析提取出的JSON数组部分
                json_str = "[" + matches.group(1) + "]"
                return parse_output(json_str)  # 递归调用解析提取出的JSON部分
            except json.JSONDecodeError:
                pass
    
    # 尝试从文本中提取电影ID（旧方法，作为备用）
    # 1. 首先尝试提取"id": X 形式的ID
    id_patterns = re.findall(r'"id"\s*:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 2. 如果没找到，尝试提取ID: X 形式的ID
    id_patterns = re.findall(r'ID:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 3. 最后尝试提取所有整数
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return [int(num) for num in numbers]
    
    # 如果没有找到任何电影ID，返回空列表
    return []