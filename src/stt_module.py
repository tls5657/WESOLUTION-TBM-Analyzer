# stt_module.py
from transformers import pipeline
import torch
import os

def transcribe_audio(audio_path, model_path):
    """긴 오디오 파일을 받아 전체 텍스트로 변환합니다."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 1단계: STT 변환 시작 (장치: {device}) ---")
    print(f"모델 로드: {model_path}")

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            device=device
        )

        print(f"음성 파일 처리 중: {os.path.basename(audio_path)}")
        result = pipe(
            audio_path,
            return_timestamps=True,
            generate_kwargs={"language": "korean", "task": "transcribe"}
        )
        
        transcribed_text = result["text"]
        print("✅ STT 변환 완료.")
        return transcribed_text

    except Exception as e:
        print(f"STT 변환 중 오류 발생: {e}")
        return None