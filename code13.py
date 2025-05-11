import json
import re

def construct_prompt(d):
    """
    Construct multi-turn dialogue prompts for large language models (chain-of-thought guidance + few-shot prompting strategy)

    Parameters:
    d (dict): A line from the jsonl data file, dictionary type variable. See val.jsonl for detailed data format

    Returns:
    list: OpenAI API message format list, simulating multi-turn conversation
    """
    user_id = d["user_id"]
    watched_movies = d["item_list"]
    watched_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in watched_movies])
    candidate_movies = d["candidates"]
    candidate_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in candidate_movies])

    # Example user-assistant dialogue (Few-shot example)
    example_user_content = (
        "User ID: 659\n\n"
        "User's watched movie list:\n"
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

        "Candidate movie list:\n"
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
        "First, I will analyze the genre categories of the user's movie history:\n\n"        
        "| Movie | Genre Categories |\n"
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
        
        "The user clearly prefers Drama films, followed by Romance. Comedy films are also somewhat preferred, but relatively less so. The user has some interest in Action and Sci-Fi films. There are no Animation, Children's, Musical, Adventure, War, Western, Fantasy, Horror, Crime, Film-Noir or Documentary films in the user's history."

        
        "Next, analyze each candidate movie's genre categories:\n\n"
        
        "| Movie | Genre Categories | Matching with User Preference |\n"
        "|------|------------|------------|\n"
        "| Touch | Romance | High |\n"
        "| Sister Act | Comedy, Crime | High |\n"
        "| Dracula | Horror | Low |\n"
        "| The Third Miracle | Drama | High |\n"
        "| Airplane! | Comedy | High |\n"
        "| La Collectionneuse | Drama | High |\n"
        "| Groove | Drama | High |\n"
        "| Truth or Consequences, N.M. | Action, Crime, Romance | Medium |\n"
        "| Century | Drama | High |\n"
        "| Naked Gun 33 1/3: The Final Insult | Comedy | High |\n"
        "| Max Dugan Returns | Comedy | High |\n"
        "| Shadowlands | Drama, Romance | High |\n"
        "| The Good Mother | Drama | High |\n"
        "| Blood In, Blood Out (a.k.a. Bound by Honor) | Crime, Drama | Medium |\n"
        "| Murder in the First | Drama, Thriller | Medium |\n"
        "| Stalker | Mystery, Sci-Fi | Low |\n"
        "| One Crazy Summer | Comedy | High |\n"
        "| Sirens | Comedy, Drama | High |\n"
        "| Passion Fish | Drama | High |\n"
        "| Anatomy of a Murder | Drama, Mystery | Medium |\n\n"

        
        "Based on the matching degree, remove the low-matching candidate movies and keep all high-matching movies for further analysis:\n\n"
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

        
        "Now enter the recommendation re-ranking stage. Please evaluate each of the remaining high-matching candidate movies (if there are less than 10 high-matching movies, also consider the medium-matching ones), combining the user preference characteristics analyzed earlier, and deeply compare each movie's matching degree with the user's preferences (considering factors such as theme, director, and actors). Then rank the movies by their matching degree, like the fine-tuning process of a recommendation system, and put the highest-matching movies first."
        "First, the user especially likes Drama films, so movies with Drama should be prioritized. Then comes Romance and Comedy, especially when these genres are combined with Drama. Additionally, the user's history includes several movies with comedic elements, such as 'Babe' and 'The American President', so pure comedies might also be good, but maybe not as good as those combined with Drama.\n"   
        "Now, let's analyze each of the remaining movies:\n"  
  
        "1. **Touch (Romance)**: Pure Romance genre, the user's history includes several romantic movies, but usually combined with Drama, such as 'Sliding Doors' and 'Ever After'. Pure Romance might be slightly weaker, but still high-matching, but maybe not as good as those with Drama.\n"  
  
        "2. **Sister Act (Comedy, Crime)**: Comedy and Crime. The user's history does not include Crime films, but Comedy is a preferred genre. However, the Crime genre might not be as suitable for the user, but since it's high-matching, it might be due to the Comedy part. But we need to consider other factors, such as the starring actress Whoopi Goldberg, who might be popular, but it's unclear if the user likes this type of comedy.\n"  
  
        "3. **Airplane! (Comedy)**: Pure Comedy, a classic comedy film. The user likes Comedy, but might prefer those with Drama combined. However, if it's a classic comedy, it might still be a good choice.\n"  
          
        "4. **La Collectionneuse (Drama)**: Drama film, directed by Éric Rohmer, French New Wave, might be more artistic. Does the user like this type of film? The user's history mostly includes American dramas, so it's unclear, but since it's Drama, it might still be prioritized.\n"  
  
        "5. **Groove (Drama)**: A drama film about electronic music parties, might be more youthful. The user's history does not include music-related films, so it's unclear if the theme matches the user's history.\n"  
  
        "6. **Naked Gun 33 1/3 (Comedy)**: Spoof comedy, the user might like it, but similar to Airplane!, is it more preferred with Drama combined?\n"  
  
        "7. **Max Dugan Returns (Comedy)**: Family comedy, the user's history includes 'Babe', which is a children's and comedy film, so it might be suitable, but we need to consider the starring actors or director.\n"  
  
        "8. **Shadowlands (Drama, Romance)**: Combines Drama and Romance, the user's history includes several films of this type, such as 'Ever After' and 'Great Expectations', so it might be very matching. The starring actors Anthony Hopkins and Debra Winger might add to its appeal.\n"  
  
        "9. **The Good Mother (Drama)**: Pure Drama, involving family conflicts, the user likes serious drama films, such as 'Philadelphia', so it might be suitable.\n"  
  
        "10. **One Crazy Summer (Comedy)**: Teen comedy, 80s style, does the user like it? It might not be as good as other comedies, but still high-matching.\n"  
  
        "11. **Sirens (Comedy, Drama)**: Comedy and Drama, combining the user's preferred genres, might be more prioritized than pure comedies. The director John Duigan and starring actor Hugh Grant might appeal to users who like romantic comedies.\n"  
  
        "12. **Passion Fish (Drama)**: A drama film with a female lead, involving rehabilitation and interpersonal relationships, the user's history includes similar serious themes, such as 'Dead Man Walking', so it might be very matching.\n"  
  
        "Next, we need to compare the themes of these movies with the user's preferences. The user clearly prefers serious drama films, especially those with emotional depth or social issues, such as 'Philadelphia' and 'In the Name of the Father'. Romantic elements usually appear as secondary genres, such as 'Sliding Doors'. Comedy-wise, the user might prefer comedies with dramatic depth, such as the comedic elements in 'The American President' combined with political drama.\n"  
  
        "So, the ranking should prioritize movies that combine Drama and Romance, followed by pure Drama, then Drama combined with other secondary genres the user likes (such as Comedy), and finally pure Comedy.\n\n"  
  
        "Specifically:\n"  
  
        "- **Shadowlands** (Drama, Romance) combines two genres the user likes, and the theme might be more profound, about the love story of author C.S. Lewis, suitable for users who like emotional depth.\n"  

        "- **Passion Fish** (Drama) involves a female lead's rehabilitation, might be emotionally rich, matching the user's preferences.\n"  
  
        "- **The Good Mother** (Drama) is a serious family drama, similar to the user's watched film 'Philadelphia'.\n"  
  
        "- **Sirens** (Comedy, Drama) combines two genres, might be more appealing to the user.\n"  
  
        "- **La Collectionneuse** (Drama) is a French art film, but does the user like this type of film? Maybe ranked lower.\n"  
  
        "- **Groove** (Drama) is a youthful film, but it's unclear if the theme matches the user's history.\n"  
  
        "- **Touch** (Romance) is pure Romance, might not be as good as those combined with Drama.\n"  
  
        "- **Sister Act** (Comedy, Crime) has a comedic part that might be suitable, but the Crime genre is not in the user's history.\n"  
  
        "- **Airplane!**, **Naked Gun**, **Max Dugan Returns**, **One Crazy Summer** are all pure comedies or mainly comedies, the user might like them, but not as prioritized as those with Drama.\n"  
  
        "Additionally, the starring actors and directors might also influence. For example, Anthony Hopkins in 'Shadowlands' might be more appealing, while 'Sirens' has Hugh Grant, does the user like romantic comedy actors? 'Airplane!' is a classic, might have broad appeal.\n"  
  
        "Considering these factors, the possible ranking is:\n"  
  
        "1. **Shadowlands** (Drama, Romance)\n"  
        "2. **Passion Fish** (Drama)\n"  
        "3. **The Good Mother** (Drama)\n"  
        "4. **Sirens** (Comedy, Drama)\n"  
        "5. **La Collectionneuse** (Drama)\n"  
        "6. **Groove** (Drama)\n"  
        "7. **Touch** (Romance)\n"  
        "8. **Airplane!** (Comedy)\n"  
        "9. **Sister Act** (Comedy, Crime)\n"  
        "10. **Max Dugan Returns** (Comedy)\n"  
        "11. **Naked Gun 33 1/3** (Comedy)\n"  
        "12. **One Crazy Summer** (Comedy)\n"  
        
        "Now I will output a list of 10 movie recommendations sorted by matching degree, in JSON format."
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


    # Define all movie genre categories
    genres = [
        # Animation and Family genres
        "Animation", "Comedy", "Children's", "Musical", 
        # Romance and Adventure genres
        "Romance", "Adventure", "Action", 
        # Thriller and War genres
        "Thriller", "Drama", "War", "Sci-Fi", 
        # Special genres
        "Western", "Fantasy", "Horror", "Crime", "Film-Noir", "Documentary"
    ]
    
    # System prompts, using a clearer format and paragraph division
    system_content = (
        "# Role:\n"
        "You are a movie recommendation system's re-ranking assistant. Your task is to recommend the most suitable movies for the user based on their movie history, by analyzing the genre categories of the user's watched movies and the candidate movies, and ranking them by their matching degree.\n"

        "# Profile:\n"
        "You have the ability to analyze movie genres and recommend movies, and can understand the user's preferences and evaluate the similarity between movies.\n"

        "# Background:\n"
        "The user provides their movie history and a list of candidate movies. You need to analyze the user's preferences and recommend the most suitable movies.\n"

        "## Goals:\n"
            "1. Analyze the genre categories of the user's movie history.\n"
            "2. Evaluate the similarity between each candidate movie and the user's preferences.\n"
            "3. Output a list of recommended movies sorted by their matching degree, in JSON format.\n"

        "## Constraints:\n"
            "- Only consider the following movie genres for analysis: Animation, Comedy, Children's, Musical, Romance, Adventure, Action, Thriller, Drama, War, Sci-Fi, Western, Fantasy, Horror, Crime, Film-Noir, Documentary\n"
            "- The final answer should be a JSON array containing the recommended movie IDs, in the format of a Python list.\n"

        "## Workflows:\n"
            "1. **User Preference Analysis**: Analyze the genre categories of the user's movie history and identify their preferences.\n"
            "2. **Candidate Movie Evaluation**: Evaluate the similarity between each candidate movie and the user's preferences, and rank them by their matching degree.\n"
            "3. **Result Output**: Output a list of recommended movies sorted by their matching degree, in JSON format."
    )
    # First round of conversation: Analyze user's movie history and genre preferences
    user_content_round1 = (

        f"User ID: {user_id}\n\n"
        f"User's watched movie list:\n{watched_movies_text}\n\n"
        f"Candidate movie list:\n{candidate_movies_text}\n\n"
        f"Movie genre categories reference: {', '.join(genres)}\n\n"
        
        "Let's analyze the user's movie history and genre preferences step by step:\n"
        "1. Analyze the genre categories of each movie in the user's history and create a genre categorization table.\n"
        "2. Analyze the user's genre preferences based on their movie history.\n"
        "3. Analyze the genre categories of each candidate movie and evaluate their similarity with the user's preferences.\n"
        "4. Now, let's enter the recommendation re-ranking stage. Please evaluate each of the remaining high-matching candidate movies (if there are less than 10 high-matching movies, also consider the medium-matching ones), combining the user preference characteristics analyzed earlier, and deeply compare each movie's matching degree with the user's preferences (considering factors such as theme, director, and actors). Then rank the movies by their matching degree, like the fine-tuning process of a recommendation system, and put the highest-matching movies first.\n"
        "5. Please output a list of recommended movies sorted by their matching degree, in JSON format.\n"

    )

   
    return [
        {"role": "system", "content": system_content},
        # Add example conversation as a few-shot example
        {"role": "user", "content": example_user_content},
        {"role": "assistant", "content": example_assistant_content},
        # Actual user's conversation
        {"role": "user", "content": user_content_round1},

    ]

def parse_output(text):
    """
    Parse the output text from the large language model to extract the re-ranked recommendation list

    Parameters:
    text (str): Output text from the LLM under the designed prompt, i.e. the assistant's last output content,
           should contain a list of movie names and IDs sorted by recommendation priority, using JSON array format

    Returns:
    list: Parsed movie ID list (Python list format, each element is an integer representing the ID), showing the re-ranked recommendation order
    Example: [1893, 3148, 111, ...]
    """
    # Parameter validation
    if not text:
        return []
        
    # Clean the text
    text = text.strip()
    
    # Try to parse the JSON array format of the movie data
    try:
        # Directly try to parse the entire text
        movies_data = json.loads(text)
        if isinstance(movies_data, list):
            # If it's a list of movie name and ID dictionaries
            if movies_data and isinstance(movies_data[0], dict) and 'id' in movies_data[0]:
                movie_ids = [movie['id'] for movie in movies_data if 'id' in movie]
                return movie_ids
            # If it's a list of movie IDs
            elif movies_data and isinstance(movies_data[0], int):
                return movies_data
            # If it's a list of movie names, need to find the corresponding IDs from the candidate movie list
            elif movies_data and isinstance(movies_data[0], str):
                # Try to read the test sample's candidate movie information
                try:
                    with open("val.jsonl", "r", encoding="utf-8") as f:
                        # Read all lines to ensure we have the test sample
                        lines = f.readlines()
                        # Try to process all available sample data
                        all_candidates = []
                        for line in lines:
                            sample = json.loads(line)
                            candidates = sample.get("candidates", [])
                            all_candidates.extend(candidates)
                        
                        # Create a movie name to ID mapping
                        name_to_id = {}
                        for movie_id, movie_name in all_candidates:
                            name_to_id[movie_name] = movie_id
                        
                        # Map the movie names to IDs
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
                return parse_output(json_str)  # Recursively call to parse the extracted JSON part
            except json.JSONDecodeError:
                pass
    
    # Try to extract the movie IDs from the text (old method, as a backup)
    # 1. First try to extract IDs in the format "id": X
    id_patterns = re.findall(r'"id"\s*:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 2. If not found, try to extract IDs in the format ID: X
    id_patterns = re.findall(r'ID:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 3. Finally try to extract all integers
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return [int(num) for num in numbers]
    
    # If no movie IDs are found, return an empty list
    return []