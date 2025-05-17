import whisper
import numpy as np
import json
from scipy.spatial.distance import cosine

model = whisper.load_model("base")

def extract_embedding(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    with torch.no_grad():
        embedding = model.encoder(mel)
    return embedding.squeeze().cpu().numpy().mean(axis=0)

def compare(reference_audio, user_audio):
    ref_vec = extract_embedding(reference_audio)
    user_vec = extract_embedding(user_audio)
    similarity = 1 - cosine(ref_vec, user_vec)
    accuracy = round(similarity * 100, 2)
    return accuracy

def transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

# 실행
accuracy = compare("audio/reference.mp3", "audio/user.mp3")
user_text = transcribe("audio/user.mp3")

# 결과 저장 (FlutterFlow에서 불러오기 쉽게 JSON으로 저장)
result = {
    "accuracy": accuracy,
    "user_text": user_text,
    "pass": accuracy >= 80  # 기준 통과 여부
}

with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("결과 저장 완료.")
