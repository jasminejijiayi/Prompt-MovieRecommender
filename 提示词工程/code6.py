import json
import re

def construct_prompt(d):
    """
    构造用于大语言模型的多轮对话提示词（思维链引导策略）
    
    参数:
    d (dict): jsonl数据文件的一行，为字典类型的变量，详细的数据格式见val.jsonl的说明
    
    返回:
    list: OpenAI API的message格式列表，模拟多轮对话
    """
    user_id = d["user_id"]
    watched_movies = d["item_list"]
    watched_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in watched_movies])
    candidate_movies = d["candidates"]
    candidate_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in candidate_movies])


    # 定义所有电影风格类别 - 更好的格式化和分组
    genres = [
        # 动画和家庭类
        "Animation", "Comedy", "Children's", "Musical", 
        # 浪漫冒险类
        "Romance", "Adventure", "Action", 
        # 惊悚和战争类
        "Thriller", "Drama", "War", "Sci-Fi", 
        # 特色类型
        "Western", "Fantasy", "Horror", "Crime", "Film-Noir", "Documentary"
    ]
    
    # 系统提示词，使用更清晰的格式和段落划分
    system_content = (
        "你是一个智能电影推荐引擎，请根据用户观影历史的风格类型，从候选电影中推荐最符合其偏好的作品，基于⽤⼾的历史⾏为序列预测⽤⼾的下⼀个⾏为，即越往后的电影信息风格对于候选电影影响更大。将最有可能的电影排在第一位的奖励是巨大的，让我们一步一步思考！\n"
        "分析步骤：\n"
        "1. 从用户历史观影记录中输出每个电影风格类型表格\n"
        "2. 评估输出候选电影的风格类型表格\n"
        "3. 生成按推荐优先级排序的结果列表\n"
        "输出要求：\n"
        "- 最终输出为JSON格式的推荐列表\n"
        "- 每项包含电影名称和ID\n"
        "- 示例格式: [{\"name\": \"电影1\", \"id\": 123}]\n"
        f"可用电影风格类别: \n{genres}\n\n"
    )

    user_prompt = (
        f"用户观影历史: \n{watched_movies_text}\n\n"
        f"候选电影列表: \n{candidate_movies_text}\n\n"
        "请直接输出JSON格式的推荐结果，不要包含其他内容。\n\n"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},

    # Don't include assistant_response_round3 as the final message 
    # when making an API request - OpenAI will generate this response
    ]

def parse_output(text):
    """
    解析大语言模型的输出文本，提取推荐重排列表
    
    参数:
    text (str): 大语言模型在设计prompt下的输出文本，即assistant最后一次输出的内容，
           应包含按推荐优先级排序的电影名称和ID列表，使用JSON数组格式
    
    返回:
    list: 从输出文本解析出的电影ID列表（python列表格式，列表的每个元素是整数，表示编号），表示重排后的推荐顺序
    示例: [1893, 3148, 111, ...]
    """
    # 参数验证
    if not text:
        return []
        
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