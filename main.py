from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Vérifier si le fichier existe avant de le charger
DATASET_PATH = "biDataset.json"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Le fichier {DATASET_PATH} est introuvable.")

# Charger les données
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Charger le modèle uniquement au premier appel
encoder_model = None
question_embeddings = None
questions = [entry["input"] for entry in data]

def load_model():
    """Charge le modèle SentenceTransformer et encode les questions."""
    global encoder_model, question_embeddings
    if encoder_model is None:
        encoder_model = SentenceTransformer("models/paraphrase-MiniLM-L6-v2")
        question_embeddings = encoder_model.encode(questions)

def retrieve_answer(query: str) -> str:
    """Trouve la réponse la plus pertinente dans le dataset."""
    load_model()
    query_embedding = encoder_model.encode(query)
    similarities = util.cos_sim(query_embedding, question_embeddings)[0].cpu().numpy()

    most_similar_index = int(np.argmax(similarities))
    similarity_score = similarities[most_similar_index]

    if similarity_score > 0.5:
        return data[most_similar_index]["output"]
    else:
        return "Je ne connais pas la réponse à cette question."

# Modèle de requête
class QuestionRequest(BaseModel):
    question: str

# Route API
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        response = retrieve_answer(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pour exécuter l'API localement
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
