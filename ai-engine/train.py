import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)

# ---------------------------------------------------------
# ★ [설정] Hugging Face Hub 및 모델 설정
# ---------------------------------------------------------
HUB_MODEL_ID = "HYPER-KJY/academy-receipt-model" # 본인 ID 확인
PUSH_TO_HUB = True

# ★ [핵심 1] 모델 초기화 (망가진 모델 버리고 원본으로 회귀)
MODEL_ID = "HYPER-KJY/academy-receipt-model"

DATASET_PATH = "./dataset/multi_receipt_train"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
LABEL_DIR = os.path.join(DATASET_PATH, "labels")

# ★ [핵심 2] T4 GPU 메모리 최적화 설정
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
EPOCHS = 20
LEARNING_RATE = 1e-5

# ---------------------------------------------------------
# 2. 데이터셋 클래스
# ---------------------------------------------------------
class ReceiptDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, max_length=768):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.max_length = max_length
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        self.task_prompt = "<s_receipt>"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        
        label_name = img_name.replace(".jpg", ".json")
        label_path = os.path.join(self.label_dir, label_name)
        
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # (1) 파일명 제거
        if "file" in label_data:
            del label_data["file"]

        # ★ [핵심 3] 학습 방해꾼 'position'(좌표) 정보 강제 삭제
        if "receipts" in label_data:
            for receipt in label_data["receipts"]:
                if "position" in receipt:
                    del receipt["position"]
            
        target_sequence = json.dumps(label_data, ensure_ascii=False)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        input_sequence = self.task_prompt + target_sequence + self.processor.tokenizer.eos_token
        
        labels = self.processor.tokenizer(
            input_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": labels.squeeze()
        }

# ---------------------------------------------------------
# 3. 학습 실행 함수
# ---------------------------------------------------------
def train():
    print("🔥 모델 준비 중...")
    
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])
    
    # ★ [핵심 4] 이미지 크기 축소 (메모리 폭발 방지: 2560 -> 1280)
    processor.image_processor.size = {"height": 1280, "width": 960}
    print(f"📉 이미지 입력 크기 조정: {processor.image_processor.size}")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receipt>")
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # 모델 설정에도 반영
    model.config.encoder.image_size = [1280, 960]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ 학습 장치: {device}")

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 오류: 데이터 폴더({IMAGE_DIR})가 없습니다.")
        return

    train_dataset = ReceiptDataset(IMAGE_DIR, LABEL_DIR, processor)
    print(f"📊 학습 데이터 수: {len(train_dataset)}장")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./result",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,
        
        # Hugging Face Hub 설정
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
        
        # ★ [핵심 5] 8-bit Optimizer 사용 (메모리 절약)
        optim="adamw_bnb_8bit" 
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,       # 중요: tokenizer 올바르게 설정
        data_collator=default_data_collator, # 중요: 데이터 꼬임 방지
    )

    print(f"🚀 학습 시작! (Hub Upload: {PUSH_TO_HUB})")
    trainer.train()

    print("💾 로컬 저장 및 Hub 업로드 중...")
    processor.save_pretrained("./result")
    
    if PUSH_TO_HUB:
        trainer.push_to_hub()
        print(f"🎉 업로드 완료! Hugging Face에서 '{HUB_MODEL_ID}'를 확인하세요.")

if __name__ == "__main__":
    train()