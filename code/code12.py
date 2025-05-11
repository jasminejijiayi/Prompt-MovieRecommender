import json
import re

def construct_prompt(d):
    """
    构造用于大语言模型的多轮对话提示词（思维链引导策略 + 少样本提示策略）
    
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
    
    # 示例用户-助手对话 (Few-shot example)
    example_user_content = (
        "用户ID: 659\n\n"
        "用户观看过的电影列表:\n"
        "- In the Name of the Father (ID: 475)\n"
        "- American History X (ID: 2329)\n"
        "- Dead Man Walking (ID: 36)\n"
        "- Thelma & Louise (ID: 3418)\n"
        "- Contact (ID: 1584)\n"
        "- Crimson Tide (ID: 161)\n"
        "- Babe (ID: 34)\n"
        "- Swing Kids (ID: 2106)\n"
        "- Miller's Crossing (ID: 1245)\n"
        "- Truman Show, The (ID: 1682)\n"
        "- The American President (ID: 11)\n"
        "- The Client (ID: 350)\n"
        "- Twelve Monkeys (ID: 32)\n"
        "- A Thousand Acres (ID: 1624)\n"
        "- Of Mice and Men (ID: 3271)\n"
        "- The Cider House Rules (ID: 3148)\n"
        "- Sliding Doors (ID: 1680)\n"
        "- Ever After: A Cinderella Story (ID: 2125)\n"
        "- Great Expectations (ID: 1735)\n"
        "- Philadelphia (ID: 508)\n\n"

        "候选电影列表:\n"
        "- Touch (ID: 1458)\n"
        "- Sister Act (ID: 3247)\n"
        "- Dracula (ID: 2644)\n"
        "- The Third Miracle (ID: 3183)\n"
        "- Airplane! (ID: 2791)\n"
        "- La Collectionneuse (ID: 823)\n"
        "- Groove (ID: 3790)\n"
        "- Truth or Consequences, N.M. (ID: 1523)\n"
        "- Century (ID: 607)\n"
        "- Naked Gun 33 1/3: The Final Insult (ID: 370)\n"
        "- Max Dugan Returns (ID: 3497)\n"
        "- Shadowlands (ID: 534)\n"
        "- The Good Mother (ID: 3449)\n"
        "- Blood In, Blood Out (a.k.a. Bound by Honor) (ID: 3761)\n"
        "- Murder in the First (ID: 280)\n"
        "- Stalker (ID: 1232)\n"
        "- One Crazy Summer (ID: 2261)\n"
        "- Sirens (ID: 537)\n"
        "- Passion Fish (ID: 1187)\n"
        "- Anatomy of a Murder (ID: 3801)\n\n"
    )
    
    example_assistant_content = (
        "首先，我将分析用户历史电影的风格类别:\n\n"        
        "| 电影 | 风格类别 |\n"
        "|------|------------|\n"
	    "| In the Name of the Father | Drama |\n"
	    "| American History X | Drama |\n"
	    "| Dead Man Walking | Drama |\n"
	    "| Thelma & Louise | Action, Drama |\n"
	    "| Contact | Drama, Sci-Fi |\n"
	    "| Crimson Tide | Drama, Thriller, War |\n"
	    "| Babe | Children's, Comedy, Drama |\n"
	    "| Swing Kids | Drama, War |\n"
	    "| Miller's Crossing | Drama |\n"
	    "| Truman Show, The | Drama |\n"
	    "| The American President | Comedy, Drama, Romance |\n"
	    "| The Client | Drama, Mystery, Thriller |\n"
	    "| Twelve Monkeys | Drama, Sci-Fi |\n"
	    "| A Thousand Acres | Drama |\n"
	    "| Of Mice and Men | Drama |\n"
	    "| The Cider House Rules | Drama |\n"
	    "| Sliding Doors | Drama, Romance |\n"
	    "| Ever After: A Cinderella Story | Drama, Romance |\n"
	    "| Great Expectations | Drama, Romance |\n"
	    "| Philadelphia | Drama |\n"
        
        "用户明显偏好剧情类(Drama)电影，其次是浪漫类(Romance)电影。喜剧类(Comedy)电影也有所偏好，但相比之下较少。用户对动作类(Action)和科幻类(Sci-Fi)电影也有一定兴趣。完全未出现动画类(Animation)、儿童类(Children's)、音乐类(Musical)、冒险类(Adventure)、战争类(War)、西部片(Western)、奇幻类(Fantasy)、恐怖类(Horror)、犯罪类(Crime)、黑色电影(Film-Noir)或纪录片类(Documentation)电影。"

        
        "接着分析每部候选电影的风格类别:\n\n"
        
        "| 电影 | 风格类别 | 与用户偏好匹配 |\n"
        "|------|------------|------------|\n"
        "| Touch | Romance | 高 |\n"
        "| Sister Act | Comedy, Crime | 高 |\n"
        "| Dracula | Horror | 低 |\n"
        "| The Third Miracle | Drama | 高 |\n"
        "| Airplane! | Comedy | 高 |\n"
        "| La Collectionneuse | Drama | 高 |\n"
        "| Groove | Drama | 高 |\n"
        "| Truth or Consequences, N.M. | Action, Crime, Romance | 中 |\n"
        "| Century | Drama | 高 |\n"
        "| Naked Gun 33 1/3: The Final Insult | Comedy | 高 |\n"
        "| Max Dugan Returns | Comedy | 高 |\n"
        "| Shadowlands | Drama, Romance | 高 |\n"
        "| The Good Mother | Drama | 高 |\n"
        "| Blood In, Blood Out (a.k.a. Bound by Honor) | Crime, Drama | 中 |\n"
        "| Murder in the First | Drama, Thriller | 中 |\n"
        "| Stalker | Mystery, Sci-Fi | 低 |\n"
        "| One Crazy Summer | Comedy | 高 |\n"
        "| Sirens | Comedy, Drama | 高 |\n"
        "| Passion Fish | Drama | 高 |\n"
        "| Anatomy of a Murder | Drama, Mystery | 中 |\n\n"

        
        "根据匹配程度去除匹配度低的候选电影，留下所有匹配度高的电影继续分析：\n\n"
        "1. Touch (ID: 1458) \n"
        "2. Sister Act (ID: 3247) \n"
        "3. Airplane! (ID: 2791) \n"
        "4. La Collectionneuse (ID: 823) \n"
        "5. Groove (ID: 3790) \n"
        "6. Naked Gun 33 1/3: The Final Insult (ID: 370) \n"
        "7. Max Dugan Returns (ID: 3497) \n"
        "8. Shadowlands (ID: 534) \n"
        "9. The Good Mother (ID: 3449) \n"
        "10. One Crazy Summer (ID: 2261) \n"
        "11. Sirens (ID: 537) \n"
        "12. Passion Fish (ID: 1187) \n\n"

        
        "现在进入推荐的重排序阶段。请对剩余匹配度高的候选电影进行逐一评估，结合之前分析得到的用户偏好特点，深入比较每部电影与用户喜好的契合程度（考虑题材、主题、导演、演员等因素）。然后根据每部电影的匹配度高低进行排序，像推荐系统的精排过程一样将匹配度最高的电影排在前面。"
        "首先，用户特别喜欢剧情片，所以带有Drama的电影应该优先考虑。然后是Romance和Comedy，尤其是当这些类型与Drama结合时可能更符合用户口味。另外，用户历史中有几部电影有喜剧元素，比如《Babe》和《The American President》，所以单独的喜剧可能也不错，但可能不如结合Drama的更好。\n"   
        "现在逐一分析剩下的电影：\n"  
  
        "1. **Touch (Romance)**: 纯浪漫类型，用户历史中有几部浪漫电影，但通常与Drama结合，比如《Sliding Doors》、《Ever After》等。纯浪漫可能稍弱，但仍有高匹配，但可能不如带Drama的。\n"  
  
        "2. **Sister Act (Comedy, Crime)**: 喜剧和犯罪。用户历史中没有犯罪类，但喜剧是用户喜欢的。不过犯罪类型可能不太符合用户习惯，但因为是高匹配，可能因为喜剧部分。但需要看其他因素，比如主演Whoopi Goldberg可能受欢迎，但不确定用户是否喜欢这种类型的喜剧。\n"  
  
        "3. **Airplane! (Comedy)**: 纯喜剧，经典搞笑片。用户喜欢喜剧，但可能更偏好有Drama结合，不过如果是经典的喜剧可能还是不错的选择。\n"  
          
        "4. **La Collectionneuse (Drama)**: 剧情片，导演Éric Rohmer，法国新浪潮，可能比较文艺。用户是否喜欢这种风格的？历史中多是美国剧情片，可能不太确定，但因为是Drama，可能还是优先。\n"  
  
        "5. **Groove (Drama)**: 关于电子音乐派对的剧情片，可能比较年轻化。用户历史中没有音乐相关，但Drama是核心，所以可能合适。\n"  
  
        "6. **Naked Gun 33 1/3 (Comedy)**: 恶搞喜剧，用户可能喜欢，但和Airplane!类似，是否更偏好有Drama的喜剧？\n"  
  
        "7. **Max Dugan Returns (Comedy)**: 家庭喜剧，用户历史中有《Babe》是儿童和喜剧，可能适合，但需要看演员或导演是否有吸引力。\n"  
  
        "8. **Shadowlands (Drama, Romance)**: 结合Drama和Romance，用户历史中有多部这种类型，比如《Ever After》、《Great Expectations》，所以可能非常契合。主演Anthony Hopkins和Debra Winger，可能加分。\n"  
  
        "9. **The Good Mother (Drama)**: 纯剧情，涉及家庭纠纷，用户喜欢严肃剧情片，如《Philadelphia》，可能合适。\n"  
  
        "10. **One Crazy Summer (Comedy)**: 青春喜剧，80年代风格，用户是否喜欢？可能不如其他喜剧，但仍有高匹配。\n"  
  
        "11. **Sirens (Comedy, Drama)**: 喜剧和剧情，结合用户喜欢的两个类型，可能比纯喜剧更优先。导演John Duigan，主演Hugh Grant，可能吸引喜欢浪漫喜剧的用户。\n"  
  
        "12. **Passion Fish (Drama)**: 女性主角的剧情片，涉及康复和人际关系，用户历史中有类似严肃题材，如《Dead Man Walking》，可能很契合。\n"  
  
        "接下来需要比较这些电影的题材和用户偏好的契合程度。用户明显偏好严肃的剧情片，尤其是带有情感深度或社会议题的，比如《Philadelphia》、《In the Name of the Father》。浪漫元素通常是作为次要类型出现，如《Sliding Doors》。喜剧方面，用户可能更喜欢有剧情深度的喜剧，比如《The American President》中的喜剧元素结合政治剧情。\n"  
  
        "所以，排名应该优先那些同时有Drama和Romance的电影，其次是纯Drama，然后是Drama加其他用户次要类型（如Comedy），最后是纯Comedy。\n\n"  
  
        "具体来看：\n"  
  
        "- **Shadowlands**（Drama, Romance）结合了两个用户喜欢的类型，且主题可能比较深刻，关于作家C.S. Lewis的爱情故事，适合喜欢情感深度的用户。\n"  

        "- **Passion Fish**（Drama）涉及女性康复，可能情感丰富，符合用户偏好。\n"  
  
        "- **The Good Mother**（Drama）严肃家庭剧情，类似用户看过的《Philadelphia》。\n"  
  
        "- **Sirens**（Comedy, Drama）两者结合，可能更接近用户喜欢的类型。\n"  
  
        "- **La Collectionneuse**（Drama）虽然法国文艺片，但用户是否喜欢？可能排名稍后。\n"  
  
        "- **Groove**（Drama）年轻化，但不确定主题是否符合用户历史。\n"  
  
        "- **Touch**（Romance）纯浪漫，可能不如结合Drama的。\n"  
  
        "- **Sister Act**（Comedy, Crime）喜剧部分可能合适，但犯罪类型用户没看过。\n"  
  
        "- **Airplane!**, **Naked Gun**, **Max Dugan Returns**, **One Crazy Summer**都是纯喜剧或主要喜剧，用户可能喜欢，但不如带Drama的优先。\n"  
  
        "此外，演员和导演也可能影响。例如，Anthony Hopkins在《Shadowlands》中可能更具吸引力，而《Sirens》有Hugh Grant，用户是否喜欢浪漫喜剧的演员？《Airplane!》是经典，可能有广泛吸引力。\n"  
  
        "综合这些因素，可能的排序是：\n"  
  
        "1. **Shadowlands**（Drama, Romance）\n"  
        "2. **Passion Fish**（Drama）\n"  
        "3. **The Good Mother**（Drama）\n"  
        "4. **Sirens**（Comedy, Drama）\n"  
        "5. **La Collectionneuse**（Drama）\n"  
        "6. **Groove**（Drama）\n"  
        "7. **Touch**（Romance）\n"  
        "8. **Airplane!**（Comedy）\n"  
        "9. **Sister Act**（Comedy, Crime）\n"  
        "10. **Max Dugan Returns**（Comedy）\n"  
        "11. **Naked Gun 33 1/3**（Comedy）\n"  
        "12. **One Crazy Summer**（Comedy）\n"  
        
        "现在我要输出10个按匹配度从高到低排序的电影推荐 JSON 列表。"
    '''[
        {"name": "Shadowlands", "id": 534},
        {"name": "Passion Fish", "id": 1187},
        {"name": "The Good Mother", "id": 3449},
        {"name": "Sirens", "id": 537},
        {"name": "La Collectionneuse", "id": 823},
        {"name": "Groove", "id": 3790},
        {"name": "Touch", "id": 1458},
        {"name": "Airplane!", "id": 2791},
        {"name": "Sister Act", "id": 3247},
        {"name": "Max Dugan Returns", "id": 3497}
    ]'''
    )


    # 定义所有电影风格类别
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
        "用户提供了历史电影列表，以及候选电影列表，电影ID没有关联关系，需要你根据用户偏好的电影风格对候选电影进行排序。\n"

        "## Goals:\n"
            "1. 分析用户历史观影记录中的电影类别喜好。\n"
            "2. 评估每个候选电影与用户偏好的相似程度。\n"
            "3. 输出按相似度排序的最终推荐电影列表（JSON格式）。\n"

        "## Constraints:\n"
            "- 只考虑以下电影风格类别进行分析: Animation, Comedy, Children's, Musical, Romance, Adventure, Action, Thriller, Drama, War, Sci-Fi, Western, Fantasy, Horror, Crime, Film-Noir, Documentary\n"
            "- 最终答案结尾包含候选列表中的电影，且以JSON数组格式输出电影ID列表，思维链要像样本助手回答\n"

        "## Workflows:\n"
            "1. **用户偏好分析**: 统计用户历史中各类别出现频次，识别用户最喜欢的类别。\n"
            "2. **候选相似度评估**: 比较每个候选电影的类别与用户偏好类别的重合程度，判断匹配度（高/中/低）。\n"
            "3. **结果输出**: 根据匹配度对候选电影进行排序，输出匹配度最高的电影名称和ID列表（JSON）。"
    )
    # 第一轮聊天：分析用户观影历史和类别偏好
    user_content_round1 = (

        f"用户ID: {user_id}\n\n"
        f"用户观看过的电影列表:\n{watched_movies_text}\n\n"
        f"候选电影列表:\n{candidate_movies_text}\n\n"
        f"电影风格类别参考: {', '.join(genres)}\n\n"
        
        "让我们一步步思考！请按照以下步骤进行分析：\n"
        "1. 分析每部电影属于哪些风格类别，并制作风格归属表格\n"
        "2. 用户的电影风格偏好分析\n"
        "3. 分析每部候选电影属于哪些风格类别，评估相似程度，去除相似度低的候选电影\n"
        "4. 现在进入推荐的重排序阶段。请对剩余的高候选电影进行逐一评估（若高候选不足10个，则同时考虑高候选和中候选），结合之前分析得到的用户偏好特点，深入比较每部电影与用户喜好的契合程度（考虑题材、主题、导演、演员等因素）。然后根据每部电影的匹配度高低进行排序，像推荐系统的精排过程一样将匹配度最高的电影排在前面。\n"
        "5. 请输出按匹配度从高到低排序的电影推荐 JSON 列表。\n"

    )

   
    return [
        {"role": "system", "content": system_content},
        # 添加示例对话作为Few-shot示例
        {"role": "user", "content": example_user_content},
        {"role": "assistant", "content": example_assistant_content},
        # 实际用户的对话
        {"role": "user", "content": user_content_round1},

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