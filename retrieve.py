import json
from sentence_transformers import SentenceTransformer, util

# Charger le fichier JSON
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Encoder les questions avec SentenceTransformer
encoder_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
questions = [entry["input"] for entry in data]
question_embeddings = encoder_model.encode(questions)

def retrieve_answer(query):
    """Trouve la réponse la plus pertinente dans le dataset."""
    query_embedding = encoder_model.encode(query)
    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    most_similar_index = similarities.argmax().item()
    
    if similarities[most_similar_index] > 0.5:
        return data[most_similar_index]["output"]
    else:
        return "Je ne connais pas la réponse à cette question."

# Test du chatbot
query = "Peux-tu me dire ce qu'est IDEM Tech ?"
print(retrieve_answer(query))
