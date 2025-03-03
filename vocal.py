import json
import io
import pygame
from gtts import gTTS
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# Charger le fichier JSON
with open("biDataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Charger le modèle depuis le stockage local
encoder_model = SentenceTransformer("models/paraphrase-MiniLM-L6-v2")

# Encoder les questions
questions = [entry["input"] for entry in data]
question_embeddings = encoder_model.encode(questions)

# Initialiser pygame pour l'audio
pygame.mixer.init()

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
    """Lit le texte dans la langue appropriée."""
    try:
        detected_lang = detect(text)  # Détecter la langue
    except:
        detected_lang = "fr"  # Par défaut en français si la détection échoue

    # Générer la voix dans la bonne langue
    tts = gTTS(text=text, lang=detected_lang, slow=False)
    
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

# Boucle d'interaction en console
while True:
    user_input = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
    if user_input.lower() == "exit":
        print("Au revoir !")
        break
    
    response = retrieve_answer(user_input)
    print("Réponse :", response)
    speak(response)
