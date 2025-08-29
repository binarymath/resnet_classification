import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import json


st.set_page_config(page_title="Classificação de Imagens - ResNet18", layout="wide")

st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(120deg, #f0f4f8 0%, #e0e7ef 100%) !important;
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
}
.small-img img {
    max-width: 140px !important;
    height: auto !important;
    border-radius: 12px;
    box-shadow: 0 2px 12px #4f8cff22;
    margin-bottom: 12px;
    border: 1.5px solid #4f8cff33;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2563eb;
    letter-spacing: 1px;
    margin-bottom: 0.5em;
    text-align: center;
    text-shadow: 0 2px 8px #4f8cff11;
}
.subtitle {
    color: #4f8cff;
    font-weight: 600;
    margin-bottom: 1.5em;
    text-align: center;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)



st.markdown('<div class="main-title">🖼️ Classificação de Imagens com ResNet18 ONNX</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Faça upload de uma imagem e veja a classe prevista pelo modelo de forma rápida e visual!</div>', unsafe_allow_html=True)

# Layout em duas colunas
col1, col2 = st.columns(2)

# Coluna da esquerda: upload
with col1:
    st.markdown('<h3 style="color:#2563eb; margin-top:0;">Envie sua imagem</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.markdown('<div class="small-img">', unsafe_allow_html=True)
        st.image(img, caption="Pré-visualização", use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Nenhuma imagem enviada ainda.")

# Coluna da direita: resultado
with col2:
    st.markdown('<h3 style="color:#2563eb; margin-top:0;">Resultado da Classificação</h3>', unsafe_allow_html=True)
    # Carregar rótulos
    with open("imagenet_labels.json", "r") as f:
        labels = json.load(f)
    # Carregar modelo
    @st.cache_resource
    def load_model():
        return ort.InferenceSession("resnet18-v2-7.onnx")
    session = load_model()
    if uploaded_file:
        img_proc = img.resize((224, 224))
        if img_proc.mode == 'RGBA':
            img_proc = img_proc.convert('RGB')
        elif img_proc.mode == 'L':
            img_proc = img_proc.convert('RGB')
        img_np = np.array(img_proc).astype(np.float32)
        img_np = img_np.transpose(2, 0, 1)
        img_np = img_np / 255.0
        img_np = np.expand_dims(img_np, axis=0)
        # Inferência
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: img_np})
        pred = np.array(output[0][0])
        idx_max = int(np.argmax(pred))
        st.success(f"Classe prevista: {labels[idx_max]} (classe {idx_max})")
        st.info(f"Confiança: {pred[idx_max]:.4f}")
    else:
        st.info("Envie uma imagem para ver o resultado.")
