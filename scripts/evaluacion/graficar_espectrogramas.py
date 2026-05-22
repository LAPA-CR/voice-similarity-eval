import os
import glob
import numpy as np

# 1. Cargar las gráficas COMPLETAS primero para evitar el lazy_loader
import matplotlib.pyplot as plt 

# 2. Cargar librosa DESPUÉS
import librosa
import librosa.display
print("1. Inicializando motor de visualización espectral...")

def graficar_comparacion(ruta_real, ruta_sintetica, identidad):
    # Cargar audios
    y_real, sr = librosa.load(ruta_real, sr=16000)
    y_sint, _ = librosa.load(ruta_sintetica, sr=16000)

    # Recortar silencios (VAD visual)
    y_real, _ = librosa.effects.trim(y_real, top_db=20)
    y_sint, _ = librosa.effects.trim(y_sint, top_db=20)

    # Calcular Espectrogramas de Mel
    S_real = librosa.feature.melspectrogram(y=y_real, sr=sr, n_mels=128)
    S_sint = librosa.feature.melspectrogram(y=y_sint, sr=sr, n_mels=128)

    # Convertir a decibeles
    S_real_db = librosa.power_to_db(S_real, ref=np.max)
    S_sint_db = librosa.power_to_db(S_sint, ref=np.max)

    # Crear la figura comparativa
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))
    
    img1 = librosa.display.specshow(S_real_db, x_axis='time', y_axis='mel', sr=sr, ax=ax[0])
    ax[0].set(title=f'Voz Real ({identidad.title()})')
    ax[0].label_outer()

    img2 = librosa.display.specshow(S_sint_db, x_axis='time', y_axis='mel', sr=sr, ax=ax[1])
    ax[1].set(title=f'Clon Sintético XTTS ({identidad.title()})')

    fig.colorbar(img1, ax=ax, format="%+2.f dB")
    
    # Guardar en la nueva estructura de carpetas
    nombre_salida = f"outputs/graficas/espectrograma_{identidad}.png"
    plt.savefig(nombre_salida)
    plt.close()
    print(f"-> Gráfica guardada: {nombre_salida}")

# Buscar audios
archivos = glob.glob("data/raw/*.wav")
reales = [f for f in archivos if "sintetico" not in f.lower() and "xtts" not in f.lower()]
sinteticos = [f for f in archivos if "sintetico" in f.lower() or "xtts" in f.lower()]

print("2. Generando mapas de calor de frecuencias (Mel-Spectrograms)...")

for sint in sinteticos:
    nombre_sint = os.path.basename(sint)
    identidad = nombre_sint.split('_')[0].lower()

    # Buscar UNA referencia real de esa persona para graficar
    refs_identidad = [r for r in reales if os.path.basename(r).lower().startswith(identidad)]
    
    if refs_identidad:
        graficar_comparacion(refs_identidad[0], sint, identidad)

print("\nProceso visualizado con éxito. Revisa la carpeta outputs/graficas/")
