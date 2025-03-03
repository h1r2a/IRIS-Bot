import os
import json
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login, snapshot_download

app = FastAPI()

# Charger le token depuis les variables d’environnement
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ ERREUR : Le token Hugging Face est manquant. Ajoutez-le dans les variables d’environnement.")

# Connexion à Hugging Face
login(token=hf_token)

# Vérifier si le fichier dataset existe
DATASET_PATH = "biDataset.json"
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"❌ ERREUR : Le fichier {DATASET_PATH} est introuvable.")

# Charger les données
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [entry["input"] for entry in data]

# Télécharger le modèle en local pour éviter les erreurs
model_path = snapshot_download(repo_id="sentence-transformers/paraphrase-MiniLM-L6-v2", token=hf_token)

# Charger le modèle une seule fois
encoder_model = SentenceTransformer(model_path)
question_embeddings = encoder_model.encode(questions)

def retrieve_answer(query: str) -> str:
    """Trouve la réponse la plus pertinente."""
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

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        response = retrieve_answer(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Lancer l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
