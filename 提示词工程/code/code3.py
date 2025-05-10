import json
import re

def construct_prompt(d):
    """
    构造用于大语言模型的多轮对话提示词（思维链引导策略）
    
    参数:
    d (dict): jsonl数据文件的一行，为字典类型的变量，详细的数据格式见val.jsonl的说明
    
    返回:
    list: OpenAI API的message格式列表，模拟多轮对话（system + user(分析) + user(只要ID列表)）
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
        # 角色定义
        "你是负责专业的电影推荐系统的重排过程,在多种推荐场景中，序列推荐是⼀种重要形式，你考虑⽤⼾⾏为的时序信息，基于⽤⼾的历史⾏为序列预测⽤⼾的下⼀个⾏为，能够捕捉⽤⼾兴趣的动态变化，提⾼推荐的时效性和精准度。"
        
        # 任务说明
        "你的任务是根据用户的观影历史，将候选电影列表重新排序，"
        "使最可能被用户喜欢的电影排在前面。"
        
        # 分析方法
        "请只考虑电影的风格类别，找出最适合用户偏好的电影。"
        "每部电影可以属于多个类别。"
        
        # 类别限制
        f"电影风格类别仅限于以下几种：{' | '.join(genres)}"
        
        # 分析要求
        "请充分利用你的知识和推理能力，先分析用户的电影类别偏好和"
        "候选电影的类别匹配度，再给出最终排序。"
        
        # 限制条件
        "为了确保推荐有效，请只从候选电影列表中进行推荐，"
        "不要添加任何列表以外的电影。"
    )

    # 第一轮聊天：分析用户观影历史和类别偏好
    user_content_round1 = (
        # 用户基本信息
        f"用户ID: {user_id}\n\n"
        f"用户观看过的电影列表:\n{watched_movies_text}\n\n"
        
        # 步骤引导
        "请按照以下步骤进行分析：\n"
        "1. 分析每部电影属于哪些风格类别，并制作风格归属表格\n"
        "2. 统计各风格类别的出现频率，并绘制风格偏好频率表格\n"
        
        # 数据输出格式要求
        f"3. 请基于这些类别计算电影类别偏好频率数据：{genres}\n"
        "4. 将类别偏好频率输出为JSON格式，请确保数组与类别顺序对应\n"
        
        # 指定输出格式
        "请在分析结果最后将用户偏好数据以JSON格式输出，格式必须为：\n"
        "```json\n{\"类别偏好\": [类别1频率, 类别2频率, ...]}\n```\n"
        "注意：这个JSON数据将用于后续分析，请确保其格式正确。"
    )

    # 第二轮聊天：分析候选电影并与用户偏好匹配
    user_content_round2 = (
        # 候选电影列表
        f"候选电影列表:\n{candidate_movies_text}\n\n"
        
        # 类别参考信息
        f"电影风格类别参考: {', '.join(genres)}\n\n"
        
        # 分析步骤
        "请按以下步骤分析候选电影：\n"
        "1. 分析每部候选电影属于哪些风格类别，并制作风格归属表格\n"
        "2. 基于上轮输出的JSON格式用户类别偏好数据，计算各候选电影与用户偏好的相似度\n"
        "3. 计算电影类别向量与用户偏好向量的相似度，并将结果排序\n"
        "4. 根据相似度对候选电影进行降序排序（相似度越高排序越靠前）\n"
    )
    
    # 第三轮聊天：要求输出最终的JSON格式的推荐列表
    user_content_round3 = (
        "请直接输出按推荐优先级排序的电影名称和ID列表，使用JSON数组格式，每个元素包含电影名称和ID。\n"
        "格式示例：[{\"name\": \"电影1\", \"id\": 123}, {\"name\": \"电影2\", \"id\": 456}]\n"
        "只输出这个JSON数组，不要包含其他解释或说明文字。"
    )

    assistant_response_round1 = "..."  # Placeholder for analysis response
    assistant_response_round2 = "..."  # Placeholder for recommendations response

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content_round1},
        {"role": "assistant", "content": assistant_response_round1},
        {"role": "user", "content": user_content_round2},
        {"role": "assistant", "content": assistant_response_round2},
        {"role": "user", "content": user_content_round3}
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