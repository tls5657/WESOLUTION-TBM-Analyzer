# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import numpy as np
import librosa
import evaluate
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from audiomentations import Compose, TimeStretch, Gain, AddGaussianNoise
from typing import List, Dict, Any
from dataclasses import dataclass

# =============================================================================
# 1. 클래스 및 함수 정의
# =============================================================================
class MySeq2SeqTrainer(Seq2SeqTrainer):
    def get_label_names(self):
        return ["labels"]

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # pad input features and attention masks
        input_features = [{"input_features": f["input_features"], "attention_mask": f["attention_mask"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # remove BOS token if present at start of each label sequence
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

    # === 설정 ===
    # [수정] 베이스 모델을 새로운 Hub ID로 변경
    BASE_MODEL_ID = r"C:\ai_work\LORA\whisper-small-komixv2"
    
    AUDIO_DIR = r"C:\ai_work\LORA\audio_sliced"
    CSV_DIR = r"C:\ai_work\LORA\splits"
    TRAIN_CSV = os.path.join(CSV_DIR, "train2.csv")
    VALID_CSV = os.path.join(CSV_DIR, "valid2.csv")
    OUTPUT_DIR = r"C:\ai_work\LORA\whisper-peft-final-komixv2_2" # 출력 폴더명 변경
    SAMPLE_RATE = 16000

    # === 데이터 로드 ===
    data_files = {"train": TRAIN_CSV, "validation": VALID_CSV}
    raw_datasets = load_dataset("csv", data_files=data_files)


        # === 데이터 증강 ===
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    fe = processor.feature_extractor
    fe.apply_spec_augment = True # 주석 해제
    fe.mask_time_prob = 0.07     # 주석 해제
    fe.mask_time_length = 10     # 주석 해제
    fe.mask_time_min_masks = 2   # 주석 해제
    fe.mask_feature_prob = 0.07  # 주석 해제
    fe.mask_feature_length = 64  # 주석 해제
    print("SpecAugment가 활성화되었습니다.")

    # === 데이터 증강 ===
    augment_transform = Compose([
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.35),
        Gain(min_gain_db=-2, max_gain_db=2, p=0.35),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.35),
    ])

    print("프로세서 설정을 로드합니다...")
    # [수정] 새로운 모델 ID로 프로세서 로드
    # processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    def process_dataset(batch, augment=False):
        path = os.path.join(AUDIO_DIR, batch["file_name"])
        try:
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            if augment:
                audio = augment_transform(samples=audio, sample_rate=sr)
            proc = processor(audio, sampling_rate=sr, return_attention_mask=True)
            return {
                "input_features": proc.input_features[0],
                "attention_mask": proc.attention_mask[0],
                "labels": processor.tokenizer(batch["transcription"]).input_ids
            }
        except Exception as e:
            print(f"오류: {path}, {e}")
            return {"input_features": None, "attention_mask": None, "labels": None}

    # === 전처리 적용 ===
    print("데이터셋 전처리를 시작합니다...")
    train_ds = raw_datasets["train"].map(
        lambda b: process_dataset(b, augment=True),
        remove_columns=raw_datasets["train"].column_names
    )
    valid_ds = raw_datasets["validation"].map(
        lambda b: process_dataset(b, augment=False),
        remove_columns=raw_datasets["validation"].column_names
    )
    train_ds = train_ds.filter(lambda x: x["input_features"] is not None)
    valid_ds = valid_ds.filter(lambda x: x["input_features"] is not None)
    print("데이터셋 전처리 완료.")

    # === GPU에 데이터 로드 ===
    print("데이터셋을 VRAM으로 이동합니다...")
    cols = ["input_features", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols, device="cuda")
    valid_ds.set_format(type="torch", columns=cols, device="cuda")
    print("VRAM 이동 완료.")
    
    # === Data Collator 설정 ===
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer = evaluate.load("wer"); cer = evaluate.load("cer")
    def compute_metrics(pred):
        p = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        l = processor.batch_decode(
            np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids),
            skip_special_tokens=True
        )
        return {"wer": wer.compute(predictions=p, references=l), "cer": cer.compute(predictions=p, references=l)}

    # =========================================================================
    # 모델 로딩 & PEFT
    # =========================================================================
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, # [수정] 새로운 모델 ID 사용
        quantization_config=bnb,
        device_map="auto"
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.3,
        bias="none"
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # === 학습 인자 & 트레이너 생성 ===
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        dataloader_num_workers=0,
        dataloader_pin_memory=True, # 원본 코드의 True 값 유지
        optim="adamw_torch_fused",
        eval_strategy="steps", eval_steps=200,
        logging_steps=100, save_steps=200, save_total_limit=3,
        learning_rate=8e-6, weight_decay=0.05,
        lr_scheduler_type="linear", warmup_steps=200,
        max_steps=2000, bf16=True,
        predict_with_generate=True,
        load_best_model_at_end=True, metric_for_best_model="cer",
        greater_is_better=False, report_to="none",
        label_names=["labels"], seed=SEED
    )
    trainer = MySeq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer
    )

    print("\n--- 파인튜닝 시작 ---")
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final-model"))
    print("--- 파인튜닝 완료 ---")

if __name__ == "__main__":
    main()