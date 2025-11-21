# ai-engine/inference.py

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
import json
import os

# 1. 저장된 모델 경로 (학습 결과물)
MODEL_PATH = "./models/receipt_model_v1"
IMAGE_PATH = "./dataset/multi_receipt_train/images/multi_receipt_00000.jpg" # 테스트할 이미지 경로

def load_model():
    print(f"📂 모델 로딩 중... ({MODEL_PATH})")
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 저장된 모델과 프로세서 불러오기
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
    processor = DonutProcessor.from_pretrained(MODEL_PATH)
    
    print(f"✅ 모델 로드 완료! (Device: {device})")
    return model, processor, device

def run_inference(model, processor, device, image_path):
    # 이미지 준비
    image = Image.open(image_path).convert("RGB")
    
    # 모델 입력 형태로 변환
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    # 프롬프트 준비 (학습 때 썼던 시작 토큰)
    task_prompt = "<s_receipt>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    # 추론 (Generate)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=768,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    # 결과 디코딩 (토큰 -> 텍스트)
    sequence = processor.batch_decode(outputs.sequences)[0]
    
    # 특수 토큰 제거 및 JSON 파싱
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # 첫 번째 <s_receipt> 제거
    
    print(f"\n🧾 [추론 결과 Raw Text]:\n{sequence}\n")
    
    try:
        # JSON 변환 시도
        result_json = processor.token2json(sequence)
        print(f"✨ [JSON 변환 성공]:")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))
        return result_json
    except Exception as e:
        print(f"⚠️ JSON 변환 실패 (텍스트로 출력): {e}")
        return sequence

if __name__ == "__main__":
    # 테스트 실행
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 오류: 모델 폴더가 없습니다. ({MODEL_PATH}) train.py를 먼저 성공시켜주세요.")
    else:
        model, processor, device = load_model()
        
        # 테스트용 이미지가 있는지 확인
        if os.path.exists(IMAGE_PATH):
            run_inference(model, processor, device, IMAGE_PATH)
        else:
            print(f"❌ 테스트 이미지를 찾을 수 없습니다: {IMAGE_PATH}")
            print("경로를 수정하거나 generate_dataset.py를 실행하세요.")