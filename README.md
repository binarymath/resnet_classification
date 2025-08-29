# 🖼️ ResNet18 ONNX - Classificação de Imagens

Este projeto demonstra como usar o modelo pré-treinado **ResNet18** no formato ONNX para classificar imagens em Python.

## 📂 Estrutura do Projeto

```
Resnet/
├── resnet18-v2-7.onnx           # Modelo pré-treinado ONNX
├── inferencia_resnet.py         # Script de inferência
├── imagenet_labels.json         # Rótulos das classes do ImageNet
├── sua_imagem.jpg               # (adicione sua imagem aqui)
```

## 🚀 Como Usar

1. **Instale as dependências:**
   ```powershell
   pip install onnxruntime numpy pillow
   ```

2. **Coloque sua imagem na pasta do projeto** (ex: `minha_imagem.jpg`).

3. **Execute o script:**
   ```powershell
   python inferencia_resnet.py
   ```

4. **Resultado:**
   O script irá mostrar a classe prevista (número e nome) para a imagem fornecida.

## 📝 Sobre o ResNet18

- ResNet18 é uma rede neural profunda muito utilizada para tarefas de classificação de imagens.
- O modelo fornecido foi treinado no conjunto de dados [ImageNet](https://www.image-net.org/), reconhecendo 1000 tipos de objetos.

## ⚠️ Observações

- O script atual classifica a imagem como um todo (não detecta múltiplos objetos).
- Para detecção de múltiplos objetos, utilize modelos como YOLO, SSD ou Faster R-CNN.

## 📚 Referências
- [ONNX Runtime](https://onnxruntime.ai/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [ImageNet Labels](https://github.com/anishathalye/imagenet-simple-labels)

---

