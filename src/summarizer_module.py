# summarizer_module.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

def summarize_text(text_to_summarize, model_id="Qwen/Qwen2-1.5B-Instruct"):
    """
    후가공된 텍스트를 받아 Qwen2 LLM으로 요약합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- 3단계: LLM 요약 시작 (모델: {model_id}, 장치: {device}) ---")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"LLM 모델 로드 중 오류 발생: {e}")
        return None

    # [수정] 아래 코드 블록 전체를 들여쓰기하여 함수 안으로 이동시켰습니다.
    # LLM이 목록 형태로 응답하도록 하는 프롬프트입니다.
# [수정] '작업 -> 위험 -> 대책'의 관계를 학습하도록 지침 강화
    messages = [
        {
            "role": "system",
            "content": "당신은 TBM 회의록을 분석하여 안전 관련 핵심 정보만 추출하는 전문가입니다. 당신의 임무는 텍스트에 나타난 각 '작업'과 그에 직접적으로 연결된 '위험 요인', '안전 대책'을 한 묶음으로 찾아내어 정리하는 것입니다. 문맥을 파악하여 STT 오류를 최대한 자연스럽게 교정해야 합니다."
        },
        {
            "role": "user",
            "content": f"""
    [지시]
    아래 [원본 텍스트]에서 논의된 '주요 작업'을 모두 찾아주세요.
    그리고 각 작업에 대해 직접적으로 언급된 '위험 요인'과 그에 대한 '안전 대책'을 찾아서, 아래 [출력 형식]에 맞춰 짝을 지어 요약해 주세요.

    [규칙]
    - 하나의 '작업'을 찾으면, 반드시 그 작업에 대한 '위험 요인'과 '안전 대책'을 같은 묶음으로 정리해야 합니다.
    - 관련 없는 정보(체조, 인사, 날씨, 잡담 등)는 철저히 무시하세요.
    - 만약 특정 작업에 대한 위험 요인이나 안전 대책이 언급되지 않았다면, 해당 항목은 '언급 없음'으로 표시하세요.
    - 다른 설명 없이, [출력 형식]에 따라 요약된 내용만 출력하세요.

    [출력 형식]
    - 작업: [첫 번째 작업 내용]
    - 위험 요인: [첫 번째 작업의 위험 요인]
    - 안전 대책: [첫 번째 작업의 안전 대책]

    - 작업: [두 번째 작업 내용]
    - 위험 요인: [두 번째 작업의 위험 요인]
    - 안전 대책: [두 번째 작업의 안전 대책]

    [원본 텍스트]
    {text_to_summarize}
    """
        }
    ]
    
    # 모델 입력 형식으로 변환
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    print("LLM이 요약을 생성 중입니다...")

    # 텍스트 생성
    generated_ids = model.generate(
        model_inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1
        repetition_penalty=1.2   # 🔁 같은 단어 반복 방지
    )

    response = tokenizer.decode(generated_ids[0, model_inputs.shape[1]:], skip_special_tokens=True)
    print("✅ LLM 요약 완료.")
    return response