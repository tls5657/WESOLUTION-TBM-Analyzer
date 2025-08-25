# -*- coding: utf-8 -*-
import os
import warnings
import torch
import whisperx  # <--- whisperx 임포트
from llama_cpp import Llama
import logging
import time
import re

# 경고 메시지 무시 및 로깅 레벨 설정
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("llama_cpp").setLevel(logging.ERROR)

# ==============================================================================
# 경로 및 설정 상수
# ==============================================================================
# ★★★ WhisperX용으로 변환된 모델 경로로 변경 ★★★
WHISPERX_MODEL_PATH = r"C:\Users\user\Desktop\LORA\stt\whisper_small_ct2" 
GGUF_MODEL_PATH     = r"C:\Users\user\Desktop\LORA\LLM\A.X-4.0-Light-Q4_K_M.gguf"
AUDIO_FILE          = r"C:\Users\user\Desktop\LORA\output.wav"

# ==============================================================================
# 1. WhisperX를 사용한 음성 텍스트 변환 (STT)
# ==============================================================================
def transcribe_with_whisperx(model_path: str, audio_file: str) -> str:
    """
    WhisperX를 사용하여 긴 오디오 파일을 매우 빠르고 정확하게 전사합니다.
    VAD, Chunking, Batching 등 모든 복잡한 과정이 라이브러리 내에서 처리됩니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "float32" 
    
    print(f"사용 디바이스: {device}, 계산 타입: {compute_type}")
    print(f"WhisperX 모델 로딩 중: {model_path}")

    # 사용자님의 커스텀 모델을 로드합니다.
    model = whisperx.load_model(
        model_path, 
        device, 
        compute_type=compute_type,
        language="ko"
    )
    print("모델 로딩 완료.")

    # whisperx.load_audio가 오디오를 알아서 전처리합니다.
    print(f"\n오디오 파일 로딩: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    
    # VRAM 크기에 따라 batch_size를 조절할 수 있습니다. (예: 8, 16, 32)
    batch_size = 16 
    print(f"전사 실행 중... (Batch size: {batch_size})")
    
    # model.transcribe() 함수 하나가 모든 것을 처리합니다.
    result = model.transcribe(audio, batch_size=batch_size)

    # 결과 텍스트만 조합하여 반환합니다.
    full_transcript = " ".join([segment['text'].strip() for segment in result['segments']])
    
    return full_transcript.strip()

# ==============================================================================
# 2. LlamaCpp를 사용한 텍스트 처리 (교정 및 요약)
# ==============================================================================
def load_llama_model(gguf_path: str) -> Llama:
    """
    GGUF 모델을 로드하며 n_ctx를 충분히 크게 설정합니다.
    """
    return Llama(model_path=gguf_path, n_ctx=16000, verbose=False, mul_mat_q=True, n_gpu_layers=-1)

def correct_and_refine_text_with_llama(text: str, model: Llama) -> str:
    """
    Whisper 전사 결과를 Llama 모델로 1차 교정 및 정제합니다. (단계별 사고 유도 버전)
    """
    prompt = f"""당신은 뛰어난 추론 능력을 가진 작업 안전 전문가입니다.
STT로 변환된 아래 녹취록에는 문맥에 맞지 않는 단어들이 포함되어 있습니다.
다음과 같은 '사고 과정'에 따라 녹취록을 완벽하게 교정해주세요.

[사고 과정]
1. **오류 식별:** STT 오류로 보이는 어색한 단어나 구절을 찾습니다.
2. **문맥 분석:** 해당 단어의 앞뒤 문장을 통해 어떤 작업이나 상황을 설명하는지 파악합니다.
3. **불필요한 중복 제거:** 의미 없이 반복되는 단어나 구절을 찾아, 문맥에 맞게 자연스러운 표현으로 다듬습니다.
4. **전문 용어 추론:** 분석된 문맥에 가장 적합한 실제 작업 현장 용어를 떠올립니다.
5. **최종 수정:** 위의 추론을 바탕으로 최종 수정된 녹취록 **전체**를 작성합니다.

**[매우 중요]**
'사고 과정'은 당신이 따라야 할 지침일 뿐, 절대 출력에 포함해서는 안 됩니다.
**오직 최종적으로 완성된 [수정된 녹취록]의 내용만** 출력해주세요.

[녹취록 원본]
{text}

[수정된 녹취록]
"""
    resp = model(prompt=prompt, max_tokens=2048, echo=False, temperature=0.2)
    return resp["choices"][0]["text"].strip()

def prepare_prompt(text: str) -> str:
    """
    Llama 모델이 오디오의 맥락을 스스로 분석하여
    다양한 위험요인을 추출하도록 유도하는 최종 프롬프트 템플릿입니다. (A.X-3.1 형식 적용)
    """
    instruction = (
        "너는 작업 현장의 안전관리 전문가야. 다음 현장 회의 녹취록을 듣고, "
        "안전을 위협하는 모든 **잠재위험요인**과 그에 대한 **구체적인 대책**을 분석해줘.\n"
        "이 분석을 기반으로 아래의 형식에 맞춰 요약해. **최종 결과물 외의 다른 출력은 일체 금지한다.**\n"
        "\n"
        "**[중요]** 요약 시 다음 지침을 반드시 따르세요:\n"
        "1. **작업내용**: 준비·보조·안전 활동은 제외하고 회의에서 언급된 실제로 수행하는 물리적인 작업만 적는다."
        #"'작업내용에는 준비·보조·안전 활동은 포함하지 않는다.\n"
        "2. **잠재위험요인**: 녹취록에 언급된 모든 잠재적 위험 요소를 찾아 나열하세요.\n"
        "3. **대책**: 각 위험 요인을 예방하거나 완화하기 위한 구체적인 조치사항을 나열하세요.\n"
        "\n"
        "**출력 형식(다른 문장 절대 금지)**:\n"
        "작업내용:\n"
        "- 항목1\n"
        "- 항목2\n"
        "잠재위험요인:\n"
        "- 항목1\n"
        "- 항목2\n"
        "대책:\n"
        "- 항목1\n"
        "- 항목2\n\n"
        "**[중요]** 최종 요약본 하나만 출력하고, 불필요한 서문이나 설명은 넣지 마세요."
    )
    
    # 지시, 입력, 응답 형식으로 프롬프트 구성
    return f"""### 지시:
{instruction}

### 입력:
{text}

### 응답:"""

def summarize_with_llama(text: str, model: Llama) -> str:
    """
    1차 교정된 텍스트를 기반으로 요약합니다.
    """
    resp = model(
        prompt=prepare_prompt(text), 
        max_tokens=1024, 
        echo=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
    )
    return resp["choices"][0]["text"].strip()

# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================
def main():
    start_time = time.time()
    
    # 1. WhisperX로 음성 전사 (STT)
    print("--- 1. WhisperX로 오디오 전사 중 ---")
    # ★★★ 기존 함수 호출을 WhisperX 함수 호출로 변경 ★★★
    transcript = transcribe_with_whisperx(WHISPERX_MODEL_PATH, AUDIO_FILE)
    print("\n=== WhisperX 전사 결과 ===")
    print(transcript)
    stt_end_time = time.time()
    print(f"STT 소요 시간: {stt_end_time - start_time:.2f} 초")

    # 2. Llama 모델 로드
    print("\n--- 2. Llama 모델 로드 중 ---")
    llama = load_llama_model(GGUF_MODEL_PATH)
    
    # 3. Llama로 1차 텍스트 교정 및 정제
    print("\n--- 3. Llama로 교정 및 정제 중 ---")
    corrected_text = correct_and_refine_text_with_llama(transcript, llama)
    print("\n=== Llama 교정 결과 ===")
    print(corrected_text)

    # 4. 최종 교정된 텍스트로 요약
    print("\n--- 4. Llama로 최종 요약 중 ---")
    final_summary = summarize_with_llama(corrected_text, llama)
    
    print("\n=== 최종 요약본 ===")
    print(final_summary)
    
    end_time = time.time()
    print(f"\n스크립트 총 실행 시간: {end_time - start_time:.2f} 초")
    
    # 최종 결과를 파일로 저장
    out_path = "final_summary.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=== WhisperX 전사 원본 ===\n")
        f.write(transcript + "\n\n")
        f.write("=== Llama 교정본 ===\n")
        f.write(corrected_text + "\n\n")
        f.write("=== 최종 요약본 ===\n")
        f.write(final_summary)
    print(f"결과가 {out_path} 에 저장되었습니다.")

if __name__ == '__main__':
    main()