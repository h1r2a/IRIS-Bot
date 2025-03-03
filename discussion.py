import json
import io
import pygame
from gtts import gTTS
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr

# Charger le fichier JSON
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Charger le modèle depuis le stockage local
encoder_model = SentenceTransformer("models/paraphrase-MiniLM-L6-v2")

# Encoder les questions
questions = [entry["input"] for entry in data]
question_embeddings = encoder_model.encode(questions)

# Initialiser pygame pour l'audio
pygame.mixer.init()

# Initialiser le recognizer pour la reconnaissance vocale
recognizer = sr.Recognizer()

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

def listen():
    """Capture l'audio depuis le microphone et le convertit en texte."""
    with sr.Microphone() as source:
        print("Dites quelque chose...")
        recognizer.adjust_for_ambient_noise(source)  # Ajuster pour le bruit ambiant
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_google(audio, language="fr-FR")
            print(f"Vous avez dit : {text}")
            return text
        except sr.UnknownValueError:
            print("Désolé, je n'ai pas compris ce que vous avez dit.")
            return None
        except sr.RequestError:
            print("Désolé, le service de reconnaissance vocale est indisponible.")
            return None

# Boucle d'interaction en console
while True:
    print("Dites 'exit' pour quitter.")
    user_input = listen()  # Utiliser la reconnaissance vocale pour capturer l'entrée

    if user_input is None:
        continue  # Si la reconnaissance a échoué, redemander une entrée

    if user_input.lower() == "exit":
        print("Au revoir !")
        break
    
    response = retrieve_answer(user_input)
    print("Réponse :", response)
    speak(response)