from sentence_transformers import SentenceTransformer
import os

# Nom du modèle
model_name = "paraphrase-MiniLM-L6-v2"
save_path = "models/paraphrase-MiniLM-L6-v2"

# Télécharger et sauvegarder le modèle
print(f"Téléchargement du modèle '{model_name}'...")
model = SentenceTransformer(model_name)

# Créer le dossier de destination s'il n'existe pas
os.makedirs(save_path, exist_ok=True)

# Sauvegarder le modèle localement
model.save_pretrained(save_path)
print(f"Modèle téléchargé et sauvegardé dans : {save_path}")
