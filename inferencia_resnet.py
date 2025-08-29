import onnxruntime as ort
import numpy as np
from PIL import Image

# Nome do arquivo da imagem que você quer testar
imagem_entrada = "minha_imagem.jpg"  # Troque pelo nome da sua imagem


# 1. Carregar e preparar a imagem
img = Image.open(imagem_entrada).resize((224, 224))  # Redimensiona para 224x224
# Garante que a imagem tenha 3 canais RGB
if img.mode == 'RGBA':
    img = img.convert('RGB')
elif img.mode == 'L':
    img = img.convert('RGB')
img = np.array(img).astype(np.float32)
img = img.transpose(2, 0, 1)  # Muda para formato (C, H, W)
img = img / 255.0  # Normaliza os valores para 0-1
img = np.expand_dims(img, axis=0)  # Adiciona dimensão do batch

# 2. Carregar o modelo ONNX
session = ort.InferenceSession("resnet18-v2-7.onnx")

# 3. Fazer a inferência
entrada_nome = session.get_inputs()[0].name
saida = session.run(None, {entrada_nome: img})

# 4. Mostrar a saída (valores para cada classe)
print("Saída do modelo:")
print(saida[0])


# Dica: Para saber a classe prevista, use np.argmax(saida[0])
classe_prevista = np.argmax(saida[0])

# Ler os rótulos do ImageNet
import json
with open("imagenet_labels.json", "r") as f:
    labels = json.load(f)

# Exibir o nome da classe prevista
nome_classe = labels[classe_prevista] if classe_prevista < len(labels) else "Desconhecido"
print(f"Classe prevista: {classe_prevista} - {nome_classe}")
