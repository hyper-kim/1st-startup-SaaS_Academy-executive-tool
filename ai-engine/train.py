import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# ---------------------------------------------------------
# ★ [설정] Hugging Face Hub 설정 (본인 ID로 수정 필수)
# ---------------------------------------------------------
HUB_MODEL_ID = "HYPER-KJY/academy-receipt-model"  # 예: "본인아이디/모델명"
PUSH_TO_HUB = True  # True로 설정하면 학습 후 자동 업로드

# 모델 이름 (베이스 모델)
MODEL_ID = "naver-clova-ix/donut-base"

# 데이터 경로 (Repo 구조 기준)
# Kaggle에서 실행 시 !cp -r ... ./dataset 명령어로 이 위치에 데이터를 두게 됩니다.
DATASET_PATH = "dataset/multi_receipt_train"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
LABEL_DIR = os.path.join(DATASET_PATH, "labels")

# 학습 설정 (Kaggle P100 GPU 기준)
# P100은 메모리가 넉넉하므로(16GB) 배치 사이즈를 조금 늘려도 됩니다.
BATCH_SIZE = 2          # 1 -> 2 (터지면 1로 줄이세요)
GRADIENT_ACCUMULATION = 4 
EPOCHS = 30             # 충분한 학습을 위해 30회 추천 (Donut은 오래 걸림)
LEARNING_RATE = 1e-5

# ---------------------------------------------------------
# 2. 데이터셋 클래스 (기존과 동일)
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

        # 학습 방해 요소 'file' 키 제거
        if "file" in label_data:
            del label_data["file"]
            
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
    
    # 1. 프로세서 로드
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])

    # 2. 모델 로드
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receipt>")
    
    # 임베딩 크기 조정
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ 학습 장치: {device}")

    # 3. 데이터셋 준비
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 오류: 데이터 폴더({IMAGE_DIR})가 없습니다.")
        print("   Kaggle: !cp -r /kaggle/input/... ./dataset 명령어로 데이터를 복사했는지 확인하세요.")
        return

    train_dataset = ReceiptDataset(IMAGE_DIR, LABEL_DIR, processor)
    print(f"📊 학습 데이터 수: {len(train_dataset)}장")

    # 4. 학습 인자 설정 (Hub 업로드 옵션 추가)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./result",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="epoch",       # 매 에폭마다 저장
        save_total_limit=2,          # 용량 관리
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=2,    # Kaggle/Linux에서는 2~4 권장
        
        # ★ Hugging Face Hub 설정 ★
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True        # 비공개 저장소 권장
    )

    # 5. 트레이너 생성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.feature_extractor, # 중요: trainer가 processor 저장하게 함
    )

    # 6. 학습 시작
    print(f"🚀 학습 시작! (Hub Upload: {PUSH_TO_HUB})")
    trainer.train()

    # 7. 최종 저장 및 업로드
    print("💾 로컬 저장 및 Hub 업로드 중...")
    
    # 프로세서도 같이 저장해야 나중에 에러가 안 남
    processor.save_pretrained("./result")
    
    if PUSH_TO_HUB:
        trainer.push_to_hub()
        print(f"🎉 업로드 완료! Hugging Face에서 '{HUB_MODEL_ID}'를 확인하세요.")
    else:
        print("🎉 학습 완료! (업로드 옵션 꺼짐)")

if __name__ == "__main__":
    train()