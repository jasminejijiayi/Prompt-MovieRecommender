import json
import re

def construct_prompt(d):
    """
    Construct multi-turn conversation prompts for large language models (Chain-of-Thought + Few-shot prompting strategies)
    
    Parameters:
    d (dict): A line from the jsonl data file, a dictionary variable, see val.jsonl for detailed data format
    
    Returns:
    list: OpenAI API message format list, simulating multi-turn conversations
    """
    user_id = d["user_id"]
    watched_movies = d["item_list"]
    watched_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in watched_movies])
    candidate_movies = d["candidates"]
    candidate_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in candidate_movies])
    
    # Example user-assistant dialogue (Few-shot example)
    example_user_content = (
        "User ID: example_user_123\n\n"
        "Movies the user has watched:\n"
        "- Liar Liar (ID: 123)\n"
        "- The Mask (ID: 456)\n"
        "- Mrs. Doubtfire (ID: 789)\n"
        "- Toy Story (ID: 246)\n"
        "- The Lion King (ID: 357)\n\n"
        
        "Candidate movies list:\n"
        "- Ace Ventura: Pet Detective (ID: 111)\n"
        "- Forrest Gump (ID: 222) \n"
        "- The Shawshank Redemption (ID: 333) \n"
        "- Aladdin (ID: 444) \n"
        "- Terminator 2: Judgment Day (ID: 555) "
    )
    
    example_assistant_content = (
        "### Analyzing User's Movie Style Preferences\n\n"
        "First, I'll analyze the genre categories of each movie:\n\n"
        
        "| Movie | Genre Categories |\n"
        "|------|------------|\n"
        "| Liar Liar | Comedy |\n"
        "| The Mask | Comedy |\n"
        "| Mrs. Doubtfire | Comedy |\n"
        "| Toy Story | Animation, Children's, Comedy |\n"
        "| The Lion King | Animation, Children's, Musical |\n\n"
        
        
        "The user clearly prefers Comedy movies, followed by Animation and Children's movies.\n\n"
        
        
        "### Analyzing the Match Between Candidate Movies and User Preferences\n\n"
        "Analyzing the genre categories of each candidate movie:\n\n"
        
        "| Movie | Genre Categories | Match with User Preferences |\n"
        "|------|------------|------------|\n"
        "| Ace Ventura: Pet Detective | Comedy | High  |\n"
        "| Forrest Gump | Comedy, Drama, Romance | Medium  |\n"
        "| The Shawshank Redemption | Drama | Low  |\n"
        "| Aladdin | Animation, Children's, Comedy | High  |\n"
        "| Terminator 2 | Action, Sci-Fi, Thriller | Low |\n\n"
        
        "Based on the matching degree, I'll remove candidates with low matching and keep 10 movies with high and medium matching for further analysis:\n\n"
        "1. Aladdin (ID: 444) \n"
        "2. Ace Ventura: Pet Detective (ID: 111) -\n"
        "3. Forrest Gump (ID: 222) \n"
        "4. The Shawshank Redemption (ID: 333)\n"
        "5. Terminator 2 (ID: 555) \n\n"
        
        "Within these 10 movies, I'll consider multiple factors such as themes, subjects, directors, actors, etc., to find the most suitable movies for the user. Here's the output of the highest matching movies with names and IDs sorted (JSON):"
        "[{\"name\": \"Aladdin\", \"id\": 444}]\n"
    )


    # Define all movie genre categories
    genres = [
        # Animation and family categories
        "Animation", "Comedy", "Children's", "Musical", 
        # Romance and adventure categories
        "Romance", "Adventure", "Action", 
        # Thriller and war categories
        "Thriller", "Drama", "War", "Sci-Fi", 
        # Special categories
        "Western", "Fantasy", "Horror", "Crime", "Film-Noir", "Documentary"
    ]
    
    # System prompt, using clearer format and paragraph division
    system_content = (
        "# Role:\n"
        "You are a reranking assistant for a movie recommendation system. Your task is to recommend works from candidate movies that best match the user's preferences based on their viewing history's style types. You should predict the user's next behavior based on their historical behavior sequence, meaning that the style of more recent movies has a greater influence on candidate movies. Let's think step by step!\n"

        "# Profile:\n"
        "You have the ability to analyze movie categories and make recommendations, understand user preferences, and evaluate similarities between movies.\n"

        "# Background:\n"
        "The user has provided a list of historical movies and a list of candidate movies. The movie IDs have no correlation, and you need to rank the candidate movies based on the user's preferred movie styles.\n"

        "## Goals:\n"
            "1. Analyze the movie category preferences in the user's viewing history.\n"
            "2. Evaluate the similarity between each candidate movie and the user's preferences.\n"
            "3. Output the final recommended movie list sorted by similarity (in JSON format).\n"

        "## Constraints:\n"
            "- Only consider the following movie genre categories for analysis: Animation, Comedy, Children's, Musical, Romance, Adventure, Action, Thriller, Drama, War, Sci-Fi, Western, Fantasy, Horror, Crime, Film-Noir, Documentary\n"
            "- The final answer should only contain movies from the candidate list and output the movie ID list in JSON array format without additional explanations.\n"

        "## Workflows:\n"
            "1. **User Preference Analysis**: Count the frequency of each category in the user's history and identify the user's favorite categories.\n"
            "2. **Candidate Similarity Evaluation**: Compare the categories of each candidate movie with the user's preferred categories to determine the matching degree (high/medium/low).\n"
            "3. **Result Output**: Sort the candidate movies based on matching degree and output the list of movie names and IDs with the highest matching (JSON)."  
    )
    # First round of chat: Analyzing user's viewing history and category preferences
    user_content_round1 = (

        # User basic information
        f"User ID: {user_id}\n\n"
        f"Movies the user has watched:\n{watched_movies_text}\n\n"
        # Candidate movies list
        f"Candidate movies list:\n{candidate_movies_text}\n\n"
        
        # Category reference information
        f"Movie genre categories reference: {', '.join(genres)}\n\n"
        
        # Step guidance
        "Let's think step by step! Please analyze according to the following steps:\n"
        "1. Analyze which genre categories each movie belongs to and create a genre attribution table\n"
        "2. Analyze the user's movie genre preferences\n"
        "3. Analyze which genre categories each candidate movie belongs to, evaluate similarity, remove those with low similarity, and keep only 10 candidates\n"
        "4. Now enter the recommendation reranking stage. Please evaluate each of the remaining 10 candidate movies one by one, combining the user preference characteristics analyzed earlier to deeply compare the compatibility of each movie with the user's preferences (considering themes, subjects, directors, actors, and other factors). Then rank them according to the matching degree of each movie, placing the movies with the highest matching degree at the front, just like the fine-sorting process of a recommendation system. Please output a JSON list of movie recommendations sorted from high to low matching degree."
        "[{\"name\": \"Aladdin\", \"id\": 444}]\n"

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
    Parse the output text of the large language model and extract the recommended reranking list
    
    Parameters:
    text (str): The output text of the large language model under the designed prompt, i.e., the content of the assistant's last output,
           should contain a list of movie names and IDs sorted by recommendation priority, using JSON array format
    
    Returns:
    list: A list of movie IDs extracted from the output text (in Python list format, each element in the list is an integer representing the ID), representing the reranked recommendation order
    Example: [1893, 3148, 111, ...]
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
                    # If unable to read the sample file, try other methods
                    pass
    except json.JSONDecodeError:
        # If parsing fails, try to find the JSON array part in the text
        pattern = r'\[(.*?)\]'
        matches = re.search(pattern, text)
        if matches:
            try:
                # Try to parse the extracted JSON array part
                json_str = "[" + matches.group(1) + "]"
                return parse_output(json_str)  # 递归调用解析提取出的JSON部分
            except json.JSONDecodeError:
                pass
    
    # Try to extract movie IDs from the text (old method, as backup)
    # 1. First try to extract IDs in the form of "id": X
    id_patterns = re.findall(r'"id"\s*:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 2. If not found, try to extract IDs in the form of ID: X
    id_patterns = re.findall(r'ID:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 3. Finally try to extract all integers
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return [int(num) for num in numbers]
    
    # If no movie IDs are found, return an empty list
    return []