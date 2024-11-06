import time
from openai import OpenAI
import json

client = OpenAI(api_key="YOUR-KEY-HERE")

file_path = 'makomori_to-label.json'

with open(file_path, 'r') as f:
    films = json.load(f)

def analyze_film(film):
    prompt = f"""
    The Mako Mori Test assesses whether female characters in films are portrayed as developed, integral parts of the story, rather than merely supporting male characters. A film that passes this test should meet the following criteria:
    It has at least one prominent female character:
    The film must feature a distinct female character with development and importance to the story, beyond a background presence or minor role.
    She has her own narrative arc:
    This character undergoes personal growth, faces a meaningful challenge, or experiences significant development. Her journey is essential to her character and the story.
    Her arc does not revolve around supporting a male character’s story:
    The female character’s development, choices, and experiences should stand independently and not exist solely to serve a male character’s journey.

    You are a film critic tasked with analyzing if each film passes the Mako Mori Test.

    Output requirements:
    - Only respond in JSON format.
    - Use `"mako-mori": 0` if the film fails the test and `"mako-mori": 1` if it passes.
    - Provide a "reason" that is concise (1-3 sentences) describing why the film passes or fails.

    Here is the reasoning process you should follow:
    First, consider whether the film has a prominent female character with development and importance to the story. Second, evaluate if she has a narrative arc that involves personal growth, challenges, or significant development. Finally, ensure that her arc does not solely support a male character's journey. Use these considerations to determine if the film passes the Mako Mori Test.

    Here is the film data:
    Title: {film["title"]}
    IMDb ID: {film["imdbID"]}
    Plot: {film["Plot"]}

    Classify the film in the category "mako-mori" as 0 (the film didn't pass the test) or 1 (the film passed the test).
    Then write a "reason" paragraph of 1 to 3 sentences describing either the arc of the female character that makes the film pass or how none of the female characters have an independent arc.

    Your answer should always and strictly follow the format in these examples:

    Example of a film that doesn't pass:
    {{
        "title": "Captain America: The Winter Soldier",
        "imdbID": "1843866",
        "Plot": "Steve Roger teams up with Black Widow to try to stop a new threat, an assassin known as the Winter Soldier.",
        "Mako Mori Test": 0,
        "Reason": "Although Black Widow is a prominent female character and has her own backstory, her role largely supports Captain America's journey throughout the film. Her development is limited and ultimately serves to aid the male protagonist’s arc. The plot remains focused on Captain America and his personal challenges, with Black Widow’s contributions secondary to his storyline."
    }}

    Example of a film that passes:
    {{
        "title": "Moana",
        "imdbID": "3521164",
        "Plot": "In ancient Polynesia, when a terrible curse incurred by the demigod Maui reaches Moana's island, she answers the Ocean's call to seek out Maui to set things right."
        "Mako Mori Test": 1,
        "Reason": "Moana’s arc centers on her journey to save her island and discover her own identity, driven by her courage and sense of duty. Her story is about self-discovery and leadership, independent of any male character's influence or romantic involvement."
    }}


    Respond strictly in the format shown, without additional text or explanation.

    """

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable film critic who provides succint yet accurate film analyses according to the Mako Mori Test."},
            {"role": "user", "content": prompt}
        ]
    )
    
    response = completion.choices[0].message.content
    return response

results = []

start_time = time.time()

with open('makomori_labeled.json', 'a') as outfile:
    for film in films:
        try:
            analysis = analyze_film(film)
            results.append(analysis)
            outfile.write(analysis + '\n')
        
        except Exception as e:
            print(f"Error processing film {film['title']} (IMDb ID: {film['imdbID']}): {e}")
            break

end_time = time.time()
for result in results:
    print(result)

total_time = end_time - start_time
print(f"Execution Time: {total_time:.2f} seconds")