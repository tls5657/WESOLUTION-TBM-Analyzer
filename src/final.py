# -*- coding: utf-8 -*-

import warnings
import re
import torch
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from llama_cpp import Llama
import logging
import time
import os
from omegaconf import OmegaConf
import torchaudio


# 경고 메시지 무시 및 로깅 레벨 설정
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("llama_cpp").setLevel(logging.ERROR)


# Silero VAD 모델 로드
VAD_REPO_DIR = r"C:\ai_work\LORA\stt\silero-vad"
vad_model, vad_utils = torch.hub.load(
    repo_or_dir=VAD_REPO_DIR,
    model='silero_vad',
    source='local',
    force_reload=False,
    onnx=False
)
get_speech_timestamps = vad_utils[0]
save_audio = vad_utils[1]
read_audio = vad_utils[2]
VADIterator = vad_utils[3]


# 경로 및 설정 상수
WHISPER_MODEL_DIR = r"C:\ai_work\LORA\stt\whisper_final_4bit"
GGUF_MODEL_PATH   = r"C:\ai_work\LORA\LLM\A.X-4.0-Light-Q3_K_M.gguf"
AUDIO_FILE        = r"C:\ai_work\LORA\youtube_audio_output2.wav"


def conservative_audio_preprocessing(audio_segment: AudioSegment) -> AudioSegment:
    """
    오디오 볼륨을 보수적으로 전처리하여 노멀라이즈합니다.
    """
    original_max = audio_segment.max_dBFS
    if original_max < -35.0:
        return normalize(audio_segment, headroom=25.0)
    elif original_max > -1.0:
        return normalize(audio_segment, headroom=6.0)
    return audio_segment


def transcribe_long_audio_hybrid(model_path: str, audio_file: str) -> str:
    """
    Silero VAD와 Whisper를 결합한 하이브리드 전사.
    긴 오디오 파일을 VAD 기반으로 효율적으로 분할하고 전사합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"사용 디바이스: {device}")
    print(f"모델 로딩 중: {model_path}")

    processor = WhisperProcessor.from_pretrained(model_path)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    except Exception:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype
        ).to(device)

    model.generation_config.language = "korean"
    model.generation_config.task = "transcribe"
    model.generation_config.repetition_penalty = 1.2
    model.generation_config.no_repeat_ngram_size = 3

    print(f"\n오디오 파일 로딩: {audio_file}")
    sound = AudioSegment.from_file(audio_file)
    print(f"원본: {len(sound)/1000:.1f}초, {sound.channels}채널, {sound.frame_rate}Hz")
    if sound.channels > 1:
        sound = sound.set_channels(1)
    
    # 1. 오디오 전처리 및 VAD 입력 준비
    processed_sound = conservative_audio_preprocessing(sound)
    array = np.array(processed_sound.get_array_of_samples()).astype(np.float32)
    array /= (32768.0 if processed_sound.sample_width == 2 else 2147483648.0)
    vad_input_audio = librosa.resample(array, orig_sr=processed_sound.frame_rate, target_sr=16000)
    vad_input_tensor = torch.from_numpy(vad_input_audio)
    
    # 2. Silero VAD로 음성 구간 탐지
    print("\n--- Silero VAD로 음성 구간 탐지 중 ---")
    speech_timestamps = get_speech_timestamps(vad_input_tensor, vad_model, sampling_rate=16000,
                                             threshold=0.35,
                                             min_speech_duration_ms=500,
                                             min_silence_duration_ms=1000)

    if not speech_timestamps:
        print("VAD가 음성 구간을 찾지 못했습니다. 전체 오디오를 단일 청크로 처리합니다.")
        speech_timestamps.append({'start': 0, 'end': len(vad_input_audio)})

    # 3. 하이브리드 청크 분할 (VAD + 최대 길이 제한)
    hybrid_chunks = []
    MAX_CHUNK_DURATION_MS = 25000
    for ts in speech_timestamps:
        start_ms = ts['start'] / 16
        end_ms = ts['end'] / 16
        current_chunk = processed_sound[start_ms:end_ms]
        
        while len(current_chunk) > 0:
            if len(current_chunk) <= MAX_CHUNK_DURATION_MS:
                hybrid_chunks.append(current_chunk)
                current_chunk = AudioSegment.empty()
            else:
                chunk_to_add = current_chunk[:MAX_CHUNK_DURATION_MS]
                hybrid_chunks.append(chunk_to_add)
                current_chunk = current_chunk[MAX_CHUNK_DURATION_MS:]

    print(f"총 {len(hybrid_chunks)}개의 하이브리드 청크로 분할")

    # 4. 청크별 Whisper 전사
    full_transcript = ""
    for i, chunk in enumerate(hybrid_chunks):
        duration = len(chunk) / 1000.0
        print(f"  - 청크 {i+1}/{len(hybrid_chunks)} 처리 중... ({duration:.1f}초)")
        
        array = np.array(chunk.get_array_of_samples()).astype(np.float32)
        array /= (32768.0 if chunk.sample_width == 2 else 2147483648.0)
        chunk_16k = librosa.resample(array, orig_sr=chunk.frame_rate, target_sr=16000)

        inputs = processor(chunk_16k, sampling_rate=16000, return_tensors="pt")
        inputs = inputs.to(device)
        if hasattr(inputs, 'input_features') and torch_dtype == torch.float16:
            inputs.input_features = inputs.input_features.to(torch_dtype)
            
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                max_length=448,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                language="korean",
                task="transcribe",
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"    결과: {transcription[:80]}{'...' if len(transcription) > 80 else ''}")
        
        full_transcript += " " + transcription

    # 5. 최종 정리
    return re.sub(r'\s+', ' ', full_transcript.strip())


def load_llama_model(gguf_path: str) -> Llama:
    """
    GGUF 모델을 로드하며 n_ctx를 충분히 크게 설정합니다.
    """
    return Llama(model_path=gguf_path, n_ctx=32768, verbose=False, n_gpu_layers=-1)


def correct_and_refine_text_with_llama(text: str, model: Llama) -> str:
    """
    Whisper 전사 결과를 Llama 모델로 1차 교정 및 정제합니다.
    """
    prompt = (
        "다음은 작업 현장 회의 녹취록입니다. "
        "문맥상 어색하거나 잘못 인식된 단어들을 작업 현장의 전문 용어로 수정하고,"
        "전체적으로 자연스러운 문장으로 내용을 요약하거나 생략하지 않고 작성해주세요.\n\n"
        "텍스트:\n"
        + text
    )
    resp = model(prompt=prompt, max_tokens=1024, echo=False)
    return resp["choices"][0]["text"].strip()


def prepare_prompt(text: str) -> str:
    """
    Llama 모델이 오디오의 맥락을 스스로 분석하여
    다양한 위험요인을 추출하도록 유도하는 최종 프롬프트 템플릿입니다.
    """
    return (
        "너는 건설 현장의 안전관리 전문가야. 다음 건설 현장 회의 녹취록을 듣고, "
        "안전을 위협하는 모든 **잠재위험요인**과 그에 대한 **구체적인 대책**을 분석해줘.\n"
        "이 분석을 기반으로 아래의 형식에 맞춰 요약해. **최종 결과물 외의 다른 출력은 일체 금지한다.**\n"
        "\n"
        "**[중요]** 요약 시 다음 지침을 반드시 따르세요:\n"
        "1. **작업내용**: 회의에서 논의된 '실제 물리적 작업'을 **핵심 과업 중심으로 간결하게 묶어서** 나열하세요. "
        "'절대 작업내용 외 잠재위험요인이나 대책에 들어갈 내용은 포함하지 마세요.\n"
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
        "**[중요]** 최종 요약본 하나만 출력하고, 불필요한 서문이나 설명은 넣지 마세요.\n\n"
        "텍스트:\n"
        + text
    )


def summarize_with_llama(text: str, model: Llama) -> str:
    """
    1차 교정된 텍스트를 기반으로 요약합니다.
    """
    # 무작위성을 제거하기 위해 temperature를 0.0으로 설정
    resp = model(
        prompt=prepare_prompt(text), 
        max_tokens=1024, 
        echo=False,
        temperature=0.0,  # 답변 일관성 유지를 위해 0.0으로 설정
        top_p=1.0,        # 모든 토큰을 고려하여 샘플링
        top_k=0,          # top_k 샘플링 비활성화
    )
    return resp["choices"][0]["text"].strip()


def main():
    start_time = time.time()
    
    # 1. Whisper로 음성 전사
    print("--- 1. Whisper로 오디오 전사 중 ---")
    # 기존 transcribe_long_audio 대신 하이브리드 함수 사용
    transcript = transcribe_long_audio_hybrid(WHISPER_MODEL_DIR, AUDIO_FILE)
    print("\n=== Whisper 전사 결과 ===")
    print(transcript)

    # 2. Llama 모델 로드
    print("\n--- 2. Llama 모델 로드 중 ---")
    llama = load_llama_model(GGUF_MODEL_PATH)
    
    # 3. Llama로 1차 텍스트 교정 및 정제
    print("\n--- 3. Llama로 텍스트 교정 및 정제 중 ---")
    corrected_text = correct_and_refine_text_with_llama(transcript, llama)
    print("\n=== Llama 교정 결과 ===")
    print(corrected_text)

    # 4. 교정된 텍스트로 요약
    print("\n--- 4. Llama로 최종 요약 중 ---")
    final_summary = summarize_with_llama(corrected_text, llama)
    
    print("\n=== 최종 요약본 ===")
    print(final_summary)
    print(f"\n스크립트 실행 시간: {time.time() - start_time:.2f} 초")


if __name__ == '__main__':
    main()