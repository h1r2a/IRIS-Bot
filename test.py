import json
import io
import pygame
from fastapi import FastAPI
from pydantic import BaseModel
from gtts import gTTS
from sentence_transformers import SentenceTransformer, util

# Charger le fichier JSON
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Charger le modèle depuis le stockage local
encoder_model = SentenceTransformer("models/paraphrase-MiniLM-L6-v2")

# Encoder les questions
questions = [entry["input"] for entry in data]
question_embeddings = encoder_model.encode(questions)

# Initialiser FastAPI
app = FastAPI()

# Initialiser pygame pour l'audio
pygame.mixer.init()

# Définir le modèle de requête
class QuestionRequest(BaseModel):
    question: str

def retrieve_answer(query):
    """Trouve la réponse la plus pertinente dans le dataset."""
    query_embedding = encoder_model.encode(query)
    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    most_similar_index = similarities.argmax().item()

    if similarities[most_similar_index] > 0.5:
        return data[most_similar_index]["output"]
    else:
        return "Je ne connais pas la réponse à cette question."

def speak(text):
    """Lit le texte directement sans sauvegarde."""
    tts = gTTS(text=text, lang="fr", slow=False)
    
    # Stocker l'audio dans un buffer en mémoire
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    # Jouer l'audio avec pygame
    pygame.mixer.music.load(fp, "mp3")
    pygame.mixer.music.play()

    # Attendre que l'audio soit terminé
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Route API pour poser une question
@app.post("/ask")
def ask_question(request: QuestionRequest):
    response = retrieve_answer(request.question)
    speak(response)  # Lecture automatique
    return {"response": response}

