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
        "# Role:\n"
        "你是一名电影推荐系统的重排助手。你的任务是根据用户观影历史的风格类型，从候选电影中推荐最符合其偏好的作品，基于⽤⼾的历史⾏为序列预测⽤⼾的下⼀个⾏为，即越往后的电影信息风格对于候选电影影响更大。让我们一步步思考！\n"

        "# Profile:\n"
        "你具备电影类别分析和推荐能力，能够理解用户的喜好，并评估电影之间的相似度。\n"

        "# Background:\n"
        "用户提供了历史电影列表（按时间顺序排列，越靠后表示越近期观看），以及候选电影列表，电影ID没有关联关系，需要你根据用户偏好的电影风格对候选电影进行排序。\n"

        "## Goals:\n"
            "1. 分析用户历史观影记录中的电影类别喜好，近期观看的电影可以赋予更高权重。\n"
            "2. 评估每个候选电影与用户偏好的相似程度。\n"
            "3. 输出按相似度排序的最终推荐电影列表（JSON格式）。\n"

        "## Constraints:\n"
            "- 只考虑以下电影风格类别进行分析: Animation, Comedy, Children's, Musical, Romance, Adventure, Action, Thriller, Drama, War, Sci-Fi, Western, Fantasy, Horror, Crime, Film-Noir, Documentary\n"
            "- 最终答案仅包含候选列表中的电影，且以JSON数组格式输出电影ID列表，不附加多余解释。\n"

        "## Workflows:\n"
            "Step 1. **用户偏好分析**: 统计用户历史中各类别出现频次，近期观看的电影可以赋予更高权重，识别用户最喜欢的类别。\n"
            "Step 2. **候选相似度评估**: 比较每个候选电影的类别与用户偏好类别的重合程度，判断匹配度（高/中/低）。\n"
            "Step 3. **结果输出**: 根据匹配度对候选电影进行排序，输出匹配度最高的电影名称和ID列表（JSON）。"
    )
    # 第一轮聊天：分析用户观影历史和类别偏好
    user_content_round1 = (
        # 用户基本信息
        f"用户ID: {user_id}\n\n"
        f"用户观看过的电影列表:\n{watched_movies_text}\n\n"

        # 步骤引导
        "### Step 1: 请按照以下步骤进行分析：\n"
        "1. 分析每部电影属于哪些风格类别，并制作风格归属表格\n"
        "2. 统计各风格类别的出现频率，近期观看的电影可以赋予更高权重，并绘制风格偏好频率表格\n"
        
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
        "### Step 2: 请按以下步骤分析候选电影：\n"
        "1. 分析每部候选电影属于哪些风格类别，并制作风格归属表格\n"
        "2. 基于上轮输出的JSON格式用户类别偏好数据，计算各候选电影与用户偏好的相似度\n"
        "3. 计算电影类别向量与用户偏好向量的相似度，并将结果排序\n"
        "4. 根据相似度对候选电影进行降序排序（相似度越高排序越靠前）\n"
    )
    
    # 第三轮聊天：要求输出最终的JSON格式的推荐列表
    user_content_round3 = (
         "### Step 3: 推荐结果输出\n"
            "请根据以上分析，从候选电影中选出最终的推荐列表，按照匹配度从高到低排序，以 JSON 数组格式输出电影名称和ID列表。\n\n"
            "要求：只输出JSON，不要添加任何解释。"
        "格式示例：[{\"name\": \"电影1\", \"id\": 123}]\n"
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