# preprocess_module.py
import re

def preprocess_text(raw_text):
    """STT로 변환된 텍스트의 오류를 교정하고 불필요한 단어 및 반복 문장을 제거합니다."""

    print("\n--- 2단계: 텍스트 후가공 시작 ---")
    
    processed_text = raw_text

    # --- 1. 간단한 단어/구문 교정 및 제거 ---
    correction_rules = {
        r"tbm": "TBM", "미터": "m", "센티": "cm", "티엘": "TL",
        r"요라할 드릴 게": "", r"고아\.": "구호.",
        r"(끄덕\s*)+": "", r"(좋아\s*)+": "",
        r"\s*(네|예)[\s.,]*": " ", r"\s*이상입니다\s*": ". ",
        r"\s*안녕하십니까\s*": " ", r"\s*수고하셨습니다\s*": ""
    }
    for pattern, replacement in correction_rules.items():
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    # --- 2. [추가] 체조 구령 제거 ---
    # "하나 둘 셋..." 과 같은 패턴을 찾아 제거합니다.
    processed_text = re.sub(r'(하나|둘|셋|넷|다섯|여섯|일곱|여덟)[\s,.]*', '', processed_text)

    # --- 3. [추가] 반복되는 문장 제거 (첫 문장만 남김) ---
    sentences = processed_text.split('.') # 마침표를 기준으로 문장 분리
    unique_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        # 문장 앞뒤 공백 제거 후, 내용이 있는 문장만 처리
        cleaned_sentence = sentence.strip()
        if cleaned_sentence:
            # 이전에 보지 않은 문장일 경우에만 추가
            if cleaned_sentence not in seen_sentences:
                unique_sentences.append(cleaned_sentence)
                seen_sentences.add(cleaned_sentence)
    
    # 중복이 제거된 문장들을 다시 합침
    processed_text = ". ".join(unique_sentences) + "."
    
    # 최종적으로 여러 개의 공백을 하나로 줄임
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    print("✅ 텍스트 후가공 완료.")
    return processed_text