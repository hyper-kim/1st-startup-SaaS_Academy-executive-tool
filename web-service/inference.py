import os
import glob
import torch
from django.conf import settings
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

# 1. 경로 설정 (result 폴더)
RESULT_DIR = os.path.join(settings.BASE_DIR, '..', 'ai-engine', 'result')
RESULT_DIR = os.path.abspath(RESULT_DIR)

checkpoints = glob.glob(os.path.join(RESULT_DIR,"checkpoint-*"))

if len(checkpoints) > 0:
    checkpoints.sort()
    MODEL_DIR = checkpoints[-1]
    print(f"🔄 자동 감지된 최신 모델: {os.path.basename(MODEL_DIR)}")
else:
    MODEL_DIR = RESULT_DIR
    print(f"⚠️ 체크포인트 폴더 없음. 기본 경로 사용: {MODEL_DIR}")
    
print(f"🔥 VisionEncoderDecoder 모델 로딩 경로: {MODEL_DIR}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # 2. 모델 로드 (VisionEncoderDecoderModel)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    # 3. 이미지 프로세서(Feature Extractor) 로드
    # (이미지를 모델이 이해하는 숫자로 변환)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)

    # 4. 텍스트 토크나이저 로드 (결과 숫자를 다시 글자로 변환)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    print(f"✅ 이미지-텍스트 생성 모델 로딩 성공! (Device: {device})")

except Exception as e:
    print(f"❌ 모델 로딩 대실패: {e}")
    model = None
    feature_extractor = None
    tokenizer = None

def run_inference(image_input):
    """
    image_input: PIL Image 객체
    """
    if model is None:
        return {"error": "모델이 로드되지 않았습니다."}

    try:
        # 
        # 1. 이미지가 만약 RGB가 아니라면 변환 (안전장치)
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        # 2. 이미지 전처리
        pixel_values = feature_extractor(images=image_input, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # 3. 텍스트 생성 (Generate)
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values,
                max_length=128,      # 생성할 문장 최대 길이
                num_beams=4,         # 빔 서치 (정확도 향상)
                early_stopping=True
            )

        # 4. 결과 디코딩 (숫자 -> 텍스트)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "status": "success",
            "result": generated_text
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}