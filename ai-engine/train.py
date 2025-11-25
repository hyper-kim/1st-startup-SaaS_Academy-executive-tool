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
    default_data_collator  # ★ [추가] 이게 있어야 에러가 안 납니다!
)

# ---------------------------------------------------------
# ★ [설정] Hugging Face Hub 설정
# ---------------------------------------------------------
HUB_MODEL_ID = "HYPER-KJY/academy-receipt-model"
PUSH_TO_HUB = True

MODEL_ID = "naver-clova-ix/donut-base"
DATASET_PATH = "dataset/multi_receipt_train"
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
LABEL_DIR = os.path.join(DATASET_PATH, "labels")

# 학습 설정
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EPOCHS = 30
LEARNING_RATE = 1e-5

# ---------------------------------------------------------
# 2. 데이터셋 클래스 (이미 패딩 처리 완료됨)
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

        if "file" in label_data:
            del label_data["file"]
            
        target_sequence = json.dumps(label_data, ensure_ascii=False)
        
        # 이미지 전처리
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # 라벨 전처리 (여기서 이미 padding="max_length"로 길이를 맞춤!)
        input_sequence = self.task_prompt + target_sequence + self.processor.tokenizer.eos_token
        
        labels = self.processor.tokenizer(
            input_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length", # ★ 이미 여기서 패딩을 다 했습니다
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

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receipt>")
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 오류: 데이터 폴더가 없습니다.")
        return

    train_dataset = ReceiptDataset(IMAGE_DIR, LABEL_DIR, processor)

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
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        
        # ★ [핵심 수정] 데이터 묶을 때 텍스트용 로직 쓰지 말고, 그냥 묶어라!
        data_collator=default_data_collator, 
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