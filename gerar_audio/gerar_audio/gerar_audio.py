from scipy.io.wavfile import write
import torch
from transformers import AutoProcessor, BarkModel

# 1. Definir o dispositivo de execução (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Utilizando dispositivo: {device}")

# 2. Carregar o processador e o modelo pré-treinado de TTS (Suno Bark)
print("⏳ Carregando o modelo de Deep Learning (Suno Bark Small)...")
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)

# 3. Definir o texto de entrada e a voz em Português
texto = "Olá! Este é um exemplo de síntese de fala utilizando modelos de aprendizado básicos em Python."
voz_preset = "v2/pt_speaker_4"  # Voz pré-configurada em Português

print("🔊 Gerando o áudio a partir do texto...")

# 4. Processar o texto e enviar para o modelo
inputs = processor(texto, voice_preset=voz_preset).to(device)

# Geração do áudio sem calcular gradientes
with torch.no_grad():
    audio_array = model.generate(**inputs)

# Converte os dados do áudio para formato NumPy
audio_array = audio_array.cpu().numpy().squeeze()

# 5. Salvar o resultado em um arquivo de áudio WAV
taxa_de_amostragem = model.generation_config.sample_rate
nome_arquivo = "audio_gerado.wav"

write(nome_arquivo, taxa_de_amostragem, audio_array)

print(f"✅ Áudio gerado e salvo com sucesso no arquivo '{nome_arquivo}'!")
