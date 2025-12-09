import os
import json
import glob
import shutil
import zipfile
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

# ==============================================================================
# 1. 설정 (Configuration)
# ==============================================================================
# ★ Hugging Face 저장소 ID (본인 것으로 수정하세요!)
HUB_MODEL_ID = "HYPER-KJY/academy-receipt-model"

# 모델 베이스 (처음부터 다시 배움)
MODEL_ID = "naver-clova-ix/donut-base"

# 학습 설정 (T4 GPU 2개 기준 최적화)
# 메모리가 부족하면 BATCH_SIZE를 1로 줄이세요.
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
EPOCHS = 10 # 충분히 학습
LEARNING_RATE = 2e-5
IMAGE_SIZE = (960, 720) # (Height, Width) - 해상도 고정

# 경로 설정 (Kaggle 환경)
WORKING_DIR = "/kaggle/working"
DATASET_DIR = "/kaggle/input/academy-dataset-with-handwriting/dataset/multi_receipt_train"
IMAGE_DIR = f"{DATASET_DIR}/images"
LABEL_DIR = f"{DATASET_DIR}/labels"

# ==============================================================================
# 2. 데이터 준비 (압축 해제)
# ==============================================================================
def prepare_data():
    if os.path.exists(IMAGE_DIR):
        print(f"✅ 데이터가 이미 준비되어 있습니다: {len(os.listdir(IMAGE_DIR))}장")
        return

    print("🔍 Input 데이터셋에서 압축 파일 찾는 중...")
    zip_path = None
    # Kaggle Input 폴더 뒤지기
    for root, dirs, files in os.walk('/kaggle/input'):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                break
        if zip_path: break
    
    if not zip_path:
        # 혹시 로컬에 있나 확인
        if os.path.exists("dataset.zip"): zip_path = "dataset.zip"
        elif os.path.exists("academy_data.zip"): zip_path = "academy_data.zip"

    if zip_path:
        print(f"📦 압축 해제 시작: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(WORKING_DIR)
        print("✅ 압축 해제 완료!")
    else:
        raise FileNotFoundError("❌ 데이터셋 zip 파일을 찾을 수 없습니다! [Add Input]을 확인하세요.")

# ==============================================================================
# 3. 데이터셋 클래스 (전처리 핵심)
# ==============================================================================
class ReceiptDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, max_length=768):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.max_length = max_length
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        self.task_prompt = "<s_receipt>"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        
        # 라벨 파일 찾기
        filename = os.path.basename(image_path)
        label_name = filename.replace(".jpg", ".json")
        label_path = os.path.join(self.label_dir, label_name)
        
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # ------------------------------------------------------------------
        # ★ [핵심전략] 불필요한 정보 삭제 (AI 뇌 용량 확보)
        # ------------------------------------------------------------------
        if "file" in label_data: del label_data["file"]
        
        if "receipts" in label_data:
            for receipt in label_data["receipts"]:
                # 1. 좌표 삭제 (이미지 인식 모델에게 좌표 예측은 너무 어려움)
                if "position" in receipt: del receipt["position"]
                
                # 2. 품목(items) 삭제 (이게 제일 중요! 이름/금액 정확도 급상승 비결)
                # 텀프 보고서용: "복잡도를 줄여 핵심 정보(금액,이름) 인식률을 높이는 Feature Selection 수행"
                if "items" in receipt: del receipt["items"]
            
        target_sequence = json.dumps(label_data, ensure_ascii=False)
        
        # 입력 처리
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

# ==============================================================================
# 4. 학습 실행
# ==============================================================================
def train():
    # 1. 데이터 준비
    prepare_data()

    print("🔥 모델 로드 중...")
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])
    
    # 해상도 강제 고정 (학습/추론 일치 필수)
    processor.image_processor.size = {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]}
    print(f"📉 이미지 크기 설정: {processor.image_processor.size}")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receipt>")
    
    # 해상도 설정 모델에도 반영
    model.config.encoder.image_size = [IMAGE_SIZE[0], IMAGE_SIZE[1]]
    
    # 토큰 추가했으니 임베딩 크기 조절
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ 학습 장치: {device}")

    # 데이터셋 연결
    train_dataset = ReceiptDataset(IMAGE_DIR, LABEL_DIR, processor)
    print(f"📊 학습 데이터 수: {len(train_dataset)}장")

    # 학습 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir="./result",
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        fp16=True,                  # 속도 향상
        gradient_checkpointing=True, # 메모리 절약
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
        
        # Hub 업로드 설정
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        hub_private_repo=True,
        
        # 메모리 최적화 옵티마이저 (bitsandbytes 필요)
        optim="adamw_bnb_8bit" 
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )

    print("🚀 학습 시작!")
    trainer.train()

    print("💾 모델 저장 및 업로드 중...")
    processor.save_pretrained("./result")
    
    # optimizer.pt 같은 거대 파일 제외하고 업로드 (속도/에러 방지)
    trainer.push_to_hub(commit_message="Training complete", blocking=True)
    print("🎉 업로드 완료!")

if __name__ == "__main__":
    train()