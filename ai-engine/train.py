# ai-engine/train.py

import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, DonutProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer

# ---------------------------------------------------------
# 1. 설정 (Config)
# ---------------------------------------------------------
# 모델 이름 (Hugging Face Hub에 있는 기본 모델)
MODEL_ID = "naver-clova-ix/donut-base"

# 데이터 경로 (아까 만든 데이터셋)
DATASET_PATH = "dataset/multi_receipt_train" # (generate_dataset.py 결과물 경로)
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
LABEL_DIR = os.path.join(DATASET_PATH, "labels")

# 학습 설정 (RTX 4060 8GB 기준)
BATCH_SIZE = 1          # VRAM 부족하면 1로 줄이세요
GRADIENT_ACCUMULATION = 8 # 2 * 4 = 8 배치 효과
EPOCHS = 5              # 데이터가 많으면 3~5번만 봐도 충분함
LEARNING_RATE = 1e-5

# ---------------------------------------------------------
# 2. 데이터셋 클래스 정의 (Dataset)
# ---------------------------------------------------------
class ReceiptDataset(Dataset):
    def __init__(self, image_dir, label_dir, processor, max_length=768):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.processor = processor
        self.max_length = max_length
        
        # 파일 목록 로드
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
        
        # 프롬프트 (모델에게 "이거 읽어줘"라고 시키는 시작 토큰)
        self.task_prompt = "<s_receipt>"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. 이미지 로드
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        
        # 2. 정답 라벨(JSON) 로드
        label_name = img_name.replace(".jpg", ".json")
        label_path = os.path.join(self.label_dir, label_name)
        
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # 💡 [수정] 학습 방해 요소인 'file' 키 제거 (중요!)
        if "file" in label_data:
            del label_data["file"]
            
        # 모델은 이제 오직 영수증 내용({"receipts": [...]})만 배웁니다.
        target_sequence = json.dumps(label_data, ensure_ascii=False)
        
        # 3. 입력(Pixel Values) 변환
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # 4. 정답(Labels) 토큰화
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
    print("🔥 모델 로드 중... (인터넷 연결 필요)")
    # 1. 프로세서(이미지 처리기 + 토크나이저) 로드
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    
    # 모델에 새로운 특수 토큰(한글 등) 추가
    # (Donut 기본 모델은 한글을 잘 알지만, receipt 관련 태그를 추가해줌)
    processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])

    # 2. 모델 로드
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_receipt>")
    
    # 토크나이저 크기가 바뀌었으므로 모델 임베딩 사이즈 조절
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # GPU 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ 학습 장치: {device} (GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'None'})")

    # 3. 데이터셋 준비
    train_dataset = ReceiptDataset(IMAGE_DIR, LABEL_DIR, processor)
    print(f"📊 학습 데이터 수: {len(train_dataset)}장")

    # 4. 학습 인자 설정
    training_args = Seq2SeqTrainingArguments(
        output_dir="./result",       # 결과 저장 경로
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        fp16=True,                   # GPU 메모리 절약 (Mixed Precision)

        gradient_checkpointing = True,

        logging_steps=10,
        save_total_limit=2,          # 모델 체크포인트 최대 2개만 저장
        remove_unused_columns=False,
        report_to="none",            # WandB 등 끄기
        dataloader_num_workers=0     # 윈도우에서는 0 권장 (에러 방지)
    )

    # 5. 트레이너 생성
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 6. 학습 시작!
    print("🚀 학습 시작! (커피 한 잔 하고 오세요)")
    trainer.train()

    # 7. 모델 저장
    print("💾 모델 저장 중...")
    save_path = "./models/receipt_model_v1"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"🎉 완료! 모델이 '{save_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # 데이터셋 폴더가 있는지 확인
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ 오류: {IMAGE_DIR} 폴더가 없습니다. generate_dataset.py를 먼저 실행하세요.")
    else:
        train()