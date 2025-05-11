import json
import re

def construct_prompt(d):
    """
    Construct multi-turn conversation prompts for large language models (chain-of-thought strategy + few-shot prompting strategy)
    
    Args:
    d (dict): A line from jsonl data file, dictionary type variable. See val.jsonl for detailed data format
    
    Returns:
    list: OpenAI API message format list, simulating multi-turn conversation
    """
    user_id = d["user_id"]
    watched_movies = d["item_list"]
    watched_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in watched_movies])
    candidate_movies = d["candidates"]
    candidate_movies_text = "\n".join([f"- {movie[1]} (ID: {movie[0]})" for movie in candidate_movies])
    
    # Example user-assistant dialog (Few-shot example)
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
        "First, I will analyze the user's historical movie genres:\n\n"        
        "| Movie | Genre |\n"
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
        
        "The user clearly prefers drama movies, followed by romance movies. Comedy movies are also preferred, but to a lesser extent. The user is also interested in action and sci-fi movies. No animation, children's, musical, adventure, western, fantasy, horror, crime, film-noir, or documentary movies are present in the user's history."

        
        "Next, analyze each candidate movie's genre:\n\n"
        
        "| Movie | Genre | Match with User Preference |\n"
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

        
        "Based on the match, remove the low-match candidate movies and leave all high-match movies for further analysis:\n\n"
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

        
        "Now enter the recommendation's re-ranking stage. Please evaluate each remaining high-match candidate movie, combining the user's preference characteristics analyzed earlier, and deeply compare each movie's matching degree with the user's preferences (considering factors such as theme, director, and actors). Then, rank the movies based on their matching degree, like the fine-tuning process of a recommendation system, and put the highest matching movie first."
        "First, the user especially likes drama movies, so movies with Drama should be prioritized. Then, Romance and Comedy, especially when these types are combined with Drama, may be more in line with the user's taste. Additionally, the user's history includes several movies with comedic elements, such as 'Babe' and 'The American President', so single comedies may also be good, but may not be as good as those combined with Drama.\n"   
        "Now, let's analyze each remaining movie:\n"  
  
        "1. **Touch (Romance)**: Pure romance type, the user's history includes several romantic movies, but usually combined with Drama, such as 'Sliding Doors' and 'Ever After'. Pure romance may be slightly weaker, but still has a high match, but may not be as good as those with Drama.\n"  
  
        "2. **Sister Act (Comedy, Crime)**: Comedy and crime. The user's history does not include crime types, but comedy is a type the user likes. However, the crime type may not be very consistent with the user's habits, but because it is a high match, it may be due to the comedy part. But need to look at other factors, such as the starring Whoopi Goldberg may be popular, but it is not certain whether the user likes this type of comedy.\n"  
  
        "3. **Airplane! (Comedy)**: Pure comedy, a classic comedy film. The user likes comedy, but may prefer comedy with drama combined, but if it is a classic comedy, it may still be a good choice.\n"  
          
        "4. **La Collectionneuse (Drama)**: Drama movie, directed by Éric Rohmer, French New Wave, may be more artistic. Does the user like this style? The history is mostly American drama movies, so it is not very certain, but because it is Drama, it may still be prioritized.\n"  
  
        "5. **Groove (Drama)**: A drama movie about electronic music parties, may be more youthful. The user's history does not include music-related movies, but Drama is the core, so it may be suitable.\n"  
  
        "6. **Naked Gun 33 1/3 (Comedy)**: Spoof comedy, the user may like it, but like Airplane!, whether the user prefers comedy with drama combined?\n"  
  
        "7. **Max Dugan Returns (Comedy)**: Family comedy, the user's history includes 'Babe', which is children's and comedy, so it may be suitable, but need to look at the starring or director whether it has appeal.\n"  
  
        "8. **Shadowlands (Drama, Romance)**: Combining Drama and Romance, the user's history includes several movies of this type, such as 'Ever After' and 'Great Expectations', so it may be very consistent. Starring Anthony Hopkins and Debra Winger, may add points.\n"  
  
        "9. **The Good Mother (Drama)**: Pure drama, involving family disputes, the user likes serious drama movies, such as 'Philadelphia', so it may be suitable.\n"  
  
        "10. **One Crazy Summer (Comedy)**: Youth comedy, 80s style, does the user like it? May not be as good as other comedies, but still has a high match.\n"  
  
        "11. **Sirens (Comedy, Drama)**: Comedy and drama, combining the two types the user likes, may be more prioritized than pure comedy. Directed by John Duigan, starring Hugh Grant, may attract users who like romantic comedies.\n"  
  
        "12. **Passion Fish (Drama)**: Female protagonist's drama movie, involving rehabilitation and interpersonal relationships, the user's history includes similar serious themes, such as 'Dead Man Walking', so it may be very consistent.\n"  
  
        "Next, we need to compare the matching degree of these movies with the user's preferences. The user clearly prefers serious drama movies, especially those with emotional depth or social issues, such as 'Philadelphia' and 'In the Name of the Father'. Romantic elements usually appear as secondary types, such as 'Sliding Doors'. In terms of comedy, the user may prefer comedies with dramatic depth, such as the comedic elements in 'The American President' combined with political drama.\n"  
  
        "So, the ranking should prioritize movies that have both Drama and Romance, followed by pure Drama, then Drama combined with other secondary types the user likes (such as Comedy), and finally pure Comedy.\n\n"  
  
        "Specifically:\n"  
  
        "- **Shadowlands** (Drama, Romance) combines two types the user likes, and the theme may be more profound, about the love story of writer C.S. Lewis, suitable for users who like emotional depth.\n"  
  
        "- **Passion Fish** (Drama) involves female rehabilitation, may be emotionally rich, consistent with the user's preferences.\n"  
  
        "- **The Good Mother** (Drama) is a serious family drama, similar to the user's watched 'Philadelphia'.\n"  
  
        "- **Sirens** (Comedy, Drama) combines the two, may be closer to the user's preferred type.\n"  
  
        "- **La Collectionneuse** (Drama) although it is a French art film, does the user like this style? May be ranked later.\n"  
  
        "- **Groove** (Drama) is youthful, but it is not certain whether the theme is consistent with the user's history.\n"  
  
        "- **Touch** (Romance) is pure romance, may not be as good as those combined with Drama.\n"  
  
        "- **Sister Act** (Comedy, Crime) the comedy part may be suitable, but the crime type the user has not watched.\n"  
  
        "- **Airplane!**, **Naked Gun**, **Max Dugan Returns**, **One Crazy Summer** are all pure comedies or mainly comedies, the user may like them, but not as prioritized as those with Drama.\n"  
  
        "In addition, the starring and director may also affect. For example, Anthony Hopkins in 'Shadowlands' may be more attractive, while 'Sirens' has Hugh Grant, does the user like romantic comedy actors?\n"  
  
        "Combining these factors, the possible ranking is:\n"  
  
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
        "But need to check if there are other factors, such as release time, ratings, etc., but the user did not provide this information, so can only be based on type and theme."
        
        "Now I will output a list of 10 movie recommendations in JSON format, sorted by matching degree."
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


    # Define all movie genres
    genres = [
        # Animation and family genres
        "Animation", "Comedy", "Children's", "Musical", 
        # Romance and adventure genres
        "Romance", "Adventure", "Action", 
        # Thriller and war genres
        "Thriller", "Drama", "War", "Sci-Fi", 
        # Special genres
        "Western", "Fantasy", "Horror", "Crime", "Film-Noir", "Documentary"
    ]
    
    # System prompts, using a clearer format and paragraph division
    system_content = (
        "# Role:\n"
        "You are a movie recommendation system's re-ranking assistant. Your task is to recommend the most suitable movies for the user based on their historical movie genres, and re-rank the candidate movies based on their matching degree with the user's preferences.\n"

        "# Profile:\n"
        "You have the ability to analyze movie genres and recommend movies, and can understand the user's preferences and evaluate the similarity between movies.\n"

        "# Background:\n"
        "The user provides their historical movie list and candidate movie list. You need to analyze the user's genre preferences and re-rank the candidate movies based on their matching degree.\n"

        "## Goals:\n"
            "1. Analyze the user's historical movie genres.\n"
            "2. Evaluate the similarity between each candidate movie and the user's preferences.\n"
            "3. Output the re-ranked movie list in JSON format.\n"

        "## Constraints:\n"
            "- Only consider the following movie genres for analysis: Animation, Comedy, Children's, Musical, Romance, Adventure, Action, Thriller, Drama, War, Sci-Fi, Western, Fantasy, Horror, Crime, Film-Noir, Documentary\n"
            "- The final answer should only include movies from the candidate list, and should be in JSON array format, without additional explanations.\n"

        "## Workflows:\n"
            "1. **User preference analysis**: Count the frequency of each genre in the user's history, and identify the user's most preferred genres.\n"
            "2. **Candidate similarity evaluation**: Compare each candidate movie's genre with the user's preferred genres, and evaluate their matching degree (high/medium/low).\n"
            "3. **Result output**: Re-rank the candidate movies based on their matching degree, and output the re-ranked movie list in JSON format."
    )
    # First round of conversation: Analyze user's historical movie genres and preferences
    user_content_round1 = (

        # User's basic information
        f"User ID: {user_id}\n\n"
        f"User's watched movie list:\n{watched_movies_text}\n\n"
        # Candidate movie list
        f"Candidate movie list:\n{candidate_movies_text}\n\n"
        
        # Genre reference information
        f"Movie genres reference: {', '.join(genres)}\n\n"
        
        # Step guidance
        "Let's analyze step by step! Please follow these steps:\n"
        "1. Analyze each movie's genres and create a genre attribution table\n"
        "2. Analyze the user's movie genre preferences\n"
        "3. Analyze each candidate movie's genres and evaluate their similarity, removing low-similarity movies and leaving only high-similarity ones\n"
        "4. Now enter the recommendation's re-ranking stage. Please evaluate each remaining high-similarity candidate movie, combining the user's preference characteristics analyzed earlier, and deeply compare each movie's matching degree with the user's preferences (considering factors such as theme, director, and actors). Then, rank the movies based on their matching degree, like the fine-tuning process of a recommendation system, and put the highest matching movie first.\n"
        "5. Please output the re-ranked movie list in JSON format.\n"
        "[{\"name\": \"Aladdin\", \"id\": 444}]\n"

    )

   
    return [
        {"role": "system", "content": system_content},
        # Add example conversation as few-shot example
        {"role": "user", "content": example_user_content},
        {"role": "assistant", "content": example_assistant_content},
        # Actual user's conversation
        {"role": "user", "content": user_content_round1},

    ]

def parse_output(text):
    """
    Parse the output text of the large language model, extracting the re-ranked movie list
    
    Args:
    text (str): The output text of the large language model, which should contain the re-ranked movie list in JSON format
    
    Returns:
    list: The extracted movie ID list (in Python list format, where each element is an integer representing the movie ID), representing the re-ranked recommendation order
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
            # If it is a list of dictionaries containing movie names and IDs
            if movies_data and isinstance(movies_data[0], dict) and 'id' in movies_data[0]:
                movie_ids = [movie['id'] for movie in movies_data if 'id' in movie]
                return movie_ids
            # If it is a list of movie IDs
            elif movies_data and isinstance(movies_data[0], int):
                return movies_data
            # If it is a list of movie names, need to look up the corresponding IDs from the candidate movie list
            elif movies_data and isinstance(movies_data[0], str):
                # Try to read the test sample's candidate movie information
                try:
                    with open("val.jsonl", "r", encoding="utf-8") as f:
                        # Read all lines to ensure we have the test sample data
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
                        
                        # Map movie names to IDs
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
    
    # Try to extract movie IDs from the text (old method, as a backup)
    # 1. First try to extract "id": X format IDs
    id_patterns = re.findall(r'"id"\s*:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 2. If not found, try to extract ID: X format IDs
    id_patterns = re.findall(r'ID:\s*(\d+)', text)
    if id_patterns:
        return [int(id_str) for id_str in id_patterns]
    
    # 3. Finally, try to extract all integers
    numbers = re.findall(r'\b\d+\b', text)
    if numbers:
        return [int(num) for num in numbers]
    
    # If no movie IDs are found, return an empty list
    return []