import os
import glob
import torch
import re
import json
from django.conf import settings
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pathlib import Path

# 전역 변수 (Lazy Loading용)
model = None
processor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_lazy():
    """
    최초 요청이 들어왔을 때 모델을 로딩합니다.
    """
    global model, processor
    
    if model is not None:
        return

    print("💤 잠자던 AI 모델을 깨우는 중... (첫 로딩)")

    # 1. 경로 탐색 (web-service 상위 -> ai-engine -> result)
    WEB_SERVICE_DIR = Path(settings.BASE_DIR)
    PROJECT_ROOT = WEB_SERVICE_DIR.parent
    AI_RESULT_DIR = PROJECT_ROOT / 'ai-engine' / 'result'
    
    print(f"📁 경로 탐색 위치: {AI_RESULT_DIR}")
    
    # 2. 체크포인트 폴더 자동 찾기
    checkpoints = glob.glob(os.path.join(str(AI_RESULT_DIR), "checkpoint-*"))
    
    if len(checkpoints) > 0:
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        MODEL_DIR = checkpoints[-1]
    else:
        MODEL_DIR = str(AI_RESULT_DIR)

    print(f"🔥 최종 AI 모델 경로: {MODEL_DIR}")

    try:
        # ---------------------------------------------------------
        # [핵심 수정] 프로세서와 모델 로딩 분리
        # ---------------------------------------------------------
        
        # 1. 프로세서는 '원본 베이스 모델'에서 가져옵니다. (설정 파일 누락 방지)
        #    만약 체크포인트에 파일이 다 있다면 MODEL_DIR에서 읽겠지만, 없으면 원본에서 읽습니다.
        try:
            processor = DonutProcessor.from_pretrained(MODEL_DIR)
        except OSError:
            print("⚠️ 체크포인트에 프로세서 설정이 없어 'naver-clova-ix/donut-base'에서 로드합니다.")
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            
            # ★ 중요: 학습 때 추가했던 특수 토큰을 똑같이 추가해줘야 함
            processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])

        # 2. 모델은 '학습된 체크포인트'에서 가져옵니다.
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
        
        # 토큰 크기 맞추기 (모델은 이미 늘어나있고, 프로세서도 방금 늘렸으므로 매칭됨)
        model.to(device)
        model.eval()
        
        print(f"✅ AI 모델 로딩 완료! (Device: {device})")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        model = None
        raise e

def run_inference(image_input):
    """
    views.py에서 호출하는 함수
    """
    if model is None:
        try:
            load_model_lazy()
        except Exception as e:
            return {"status": "error", "message": f"모델 로딩 실패: {str(e)}"}

    try:
        # 1. 이미지 포맷 통일
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        # 2. 전처리
        pixel_values = processor(image_input, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # 3. 프롬프트 준비
        task_prompt = "<s_receipt>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        # 4. 생성 (Inference)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                
                # [수정된 부분] ----------------------------------
                num_beams=4,          # 1 -> 4 (더 여러 경우의 수를 탐색하여 정확도 향상)
                repetition_penalty=1.2, # 반복해서 말하면 패널티 부여 (앵무새 방지)
                no_repeat_ngram_size=3, # 3단어 이상 똑같이 반복 금지
                # -----------------------------------------------
                
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # 5. 후처리
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        print(f"🤖 AI 분석 결과: {sequence}")

        try:
            json_output = processor.token2json(sequence)
            return {"status": "success", "result": json_output}
        except:
            return {"status": "partial_success", "result": {"text_content": sequence}}

    except Exception as e:
        return {"status": "error", "message": str(e)}