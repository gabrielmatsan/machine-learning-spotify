"""
Script para fazer upload do dataset para o Modal Volume.

Uso: python upload_dataset.py
"""

import subprocess
import os

# Caminho local do dataset
LOCAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_with_features.csv")
VOLUME_NAME = "spotify-model-volume"


def main():
    if not os.path.exists(LOCAL_DATASET_PATH):
        print(f"❌ Dataset não encontrado em: {LOCAL_DATASET_PATH}")
        print("Certifique-se de que o arquivo dataset.csv está na pasta do projeto.")
        return

    size_mb = os.path.getsize(LOCAL_DATASET_PATH) / (1024 * 1024)
    print(f"📤 Enviando dataset: {LOCAL_DATASET_PATH}")
    print(f"Tamanho: {size_mb:.2f} MB")

    # Usar CLI do Modal para upload
    cmd = ["modal", "volume", "put", VOLUME_NAME, LOCAL_DATASET_PATH, "dataset.csv"]
    print(f"Executando: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Dataset enviado com sucesso para o Modal Volume!")
        print(result.stdout)
    else:
        print("❌ Erro ao enviar dataset:")
        print(result.stderr)


if __name__ == "__main__":
    main()
