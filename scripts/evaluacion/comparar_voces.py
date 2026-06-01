import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.pretrained import SpeakerRecognition
import os

# 1. Configuración de hardware 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"--- Ejecutando en: {device.upper()} ---")

# 2. Carga del modelo ECAPA-TDNN (192-dimensional embeddings)
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="models/ecapa",
    run_opts={"device": device}
)

def obtener_embedding(path):
    # Carga y normaliza el audio
    signal = model.load_audio(path).to(device)
    # Extrae el vector de identidad (embedding)
    with torch.no_grad():
        embedding = model.encode_batch(signal)
    return embedding

def test_similitud(path_ref, path_test):
    emb_ref = obtener_embedding(path_ref)
    emb_test = obtener_embedding(path_test)
    
    # Cálculo de Similitud Coseno
    score = F.cosine_similarity(emb_ref, emb_test).mean().item()
    return score

# PRUEBAS DE VALIDACIÓN
folder = "data/raw/"
ref_maestra = folder + "daniel_ref1.wav"

# Lista de archivos para comparar contra la referencia maestra
pruebas = [
    "daniel_test1.wav", 
    "daniel_test2.wav", 
    "felipe_test1.wav", 
    "felipe_test2.wav"
]

print(f"\nComparando contra referencia: {ref_maestra}")
print("-" * 50)

for p in pruebas:
    path_p = folder + p
    if os.path.exists(path_p):
        similitud = test_similitud(ref_maestra, path_p)
        resultado = "IDENTIDAD CONFIRMADA" if similitud > 0.5 else "ACCESO DENEGADO"
        print(f"Test: {p:18} | Similitud: {similitud:.4f} | {resultado}")
