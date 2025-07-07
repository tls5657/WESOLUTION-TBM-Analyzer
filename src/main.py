# main.py
import os
from stt_module import transcribe_audio
from preprocess_module import preprocess_text
from summarizer_module import summarize_text
# import json # JSON을 사용하지 않으므로 주석 처리하거나 삭제 가능

# ==============================================================================
#                               ** 설정 **
# ==============================================================================
# 1. 변환할 원본 오디오 파일 경로
ORIGINAL_AUDIO_PATH = r"C:\Users\user\Desktop\wesolution\youtube_audio_output.wav"

# 2. 파인튜닝된 Whisper 모델이 저장된 폴더 경로
WHISPER_MODEL_PATH = r"C:\Users\user\Desktop\wesolution\TBM_project\whisper-small-tbm-augmented"

# 3. 사용할 LLM 모델 ID
LLM_MODEL_ID = "Qwen/Qwen2-7B-Instruct"

# 4. 중간 결과 및 최종 결과가 저장될 파일 이름
RAW_TRANSCRIPT_FILE = "raw_transcript.txt"
CLEANED_TRANSCRIPT_FILE = "cleaned_transcript.txt"
FINAL_SUMMARY_FILE = "final_summary.txt"
# ==============================================================================

if __name__ == '__main__':
    # --- 1단계: STT 변환 실행 ---
    raw_text = transcribe_audio(ORIGINAL_AUDIO_PATH, WHISPER_MODEL_PATH)

    if raw_text:
        with open(RAW_TRANSCRIPT_FILE, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        print(f"-> 원본 텍스트가 '{RAW_TRANSCRIPT_FILE}'에 저장되었습니다.")

        # --- 2단계: 텍스트 후가공 실행 ---
        cleaned_text = preprocess_text(raw_text)
        
        with open(CLEANED_TRANSCRIPT_FILE, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"-> 후가공된 텍스트가 '{CLEANED_TRANSCRIPT_FILE}'에 저장되었습니다.")

        # --- 3단계: LLM 요약 실행 ---
        summary_output = summarize_text(cleaned_text, LLM_MODEL_ID)

        if summary_output:
            print("\n--- 최종 요약 결과 ---")
            print(summary_output) # LLM이 반환한 텍스트를 그대로 출력

            # 최종 요약을 .txt 파일로 저장
            with open(FINAL_SUMMARY_FILE, 'w', encoding='utf-8') as f:
                f.write(summary_output) # 그대로 저장
            print(f"\n✅ 모든 작업 완료! 최종 요약이 '{FINAL_SUMMARY_FILE}'에 저장되었습니다.")