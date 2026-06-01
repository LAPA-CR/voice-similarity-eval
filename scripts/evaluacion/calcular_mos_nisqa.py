import os
import subprocess
import pandas as pd
import numpy as np

print("1. Verificando motor neuronal NISQA...")
if not os.path.exists("NISQA/run_predict.py"):
    print("Error: No se encontró el motor NISQA. Ejecuta: git clone https://github.com/gabrielmittag/NISQA.git")
    exit()

print("2. Evaluando calidad y naturalidad (MOS) de los audios...")
print("   (Esto puede tomar unos segundos mientras la red neuronal procesa los espectrogramas)")

os.makedirs("resultados_nisqa", exist_ok=True)

# Comando para ejecutar NISQA en la carpeta data/raw
comando = [
    "python", "NISQA/run_predict.py",
    "--mode", "predict_dir",
    "--pretrained_model", "NISQA/weights/nisqa.tar",
    "--data_dir", "data/raw",
    "--num_workers", "0",
    "--bs", "1",
    "--output_dir", "resultados_nisqa"
]

# Ejecutar el modelo
resultado = subprocess.run(comando, capture_output=True, text=True)

if resultado.returncode != 0:
    print("\nError interno al ejecutar NISQA:")
    print(resultado.stderr)
    exit()

# 3. Leer y procesar los resultados generados por el modelo
archivo_resultados = "resultados_nisqa/NISQA_results.csv"
if not os.path.exists(archivo_resultados):
    print("\nError: NISQA no generó el archivo de resultados.")
    exit()

df = pd.read_csv(archivo_resultados)
mos_reales = []
mos_sinteticos = []

import os
import subprocess
import pandas as pd
import numpy as np

print("1. Verificando motor neuronal NISQA...")
if not os.path.exists("NISQA/run_predict.py"):
    print("Error: No se encontró el motor NISQA. Ejecuta: git clone https://github.com/gabrielmittag/NISQA.git")
    exit()

print("2. Evaluando calidad y naturalidad (MOS) de los audios...")
print("   (Esto puede tomar unos segundos mientras la red neuronal procesa los espectrogramas)")

# Comando para ejecutar NISQA en la carpeta data/raw
comando = [
    "python", "NISQA/run_predict.py",
    "--mode", "predict_dir",
    "--pretrained_model", "NISQA/weights/nisqa.tar",
    "--data_dir", "data/raw",
    "--num_workers", "0",
    "--bs", "1",
    "--output_dir", "resultados_nisqa"
]

# Ejecutar el modelo
resultado = subprocess.run(comando, capture_output=True, text=True)

if resultado.returncode != 0:
    print("\nError interno al ejecutar NISQA:")
    print(resultado.stderr)
    exit()

# 3. Leer y procesar los resultados generados por el modelo
archivo_resultados = "resultados_nisqa/NISQA_results.csv"
if not os.path.exists(archivo_resultados):
    print("\nError: NISQA no generó el archivo de resultados.")
    exit()

df = pd.read_csv(archivo_resultados)
mos_reales = []
mos_sinteticos = []

print("\n--- RESULTADOS INDIVIDUALES DE NATURALIDAD (MOS) ---")
for index, row in df.iterrows():
    archivo = row['deg']
    mos = row['mos_pred']
    
    if "sintetico" in archivo.lower() or "xtts" in archivo.lower():
        mos_sinteticos.append(mos)
        tipo = "Clon Sintético"
    else:
        mos_reales.append(mos)
        tipo = "Voz Real      "
        
    print(f"[{tipo}] {archivo}: {mos:.2f} / 5.00")

print("\n--- RESUMEN ESTADÍSTICO DE CALIDAD (NISQA) ---")
if mos_reales:
    print(f"MOS Promedio (Voces Reales): {np.mean(mos_reales):.2f} / 5.00")
if mos_sinteticos:
    print(f"MOS Promedio (Clones XTTS):  {np.mean(mos_sinteticos):.2f} / 5.00")
    
print("\n* Criterio: Escala de 1 (Malo/Robótico) a 5 (Excelente/Natural).")
print("* Nota: Los audios reales deberían acercarse a 4.0-5.0. Clones sobre 3.5 se consideran de alta calidad.")
