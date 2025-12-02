import torch
import re
import json
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

MODEL_ID = "HYPER-KJY/academy-receipt-model"

# 전역 변수
model = None
processor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_lazy():
    global model, processor
    if model is not None: return

    print(f"💤 Hugging Face Hub에서 모델 로딩 중... (ID: {MODEL_ID})")

    try:
        # 1. 프로세서 로드 (설정 파일 누락 대비 Fallback)
        try:
            processor = DonutProcessor.from_pretrained(MODEL_ID)
        except OSError:
            print("⚠️ 프로세서 설정이 없어 기본값(donut-base)을 사용합니다.")
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])

        # ★ [핵심 수정] 추론할 때도 학습 때와 똑같은 해상도로 강제 고정! ★
        # 이 코드가 없으면 모델이 이미지를 2배 크게(잘못) 봅니다.
        processor.image_processor.size = {"height": 1280, "width": 960}
        print(f"📉 추론 이미지 크기 조정: {processor.image_processor.size}")

        # 2. 모델 로드
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        # 모델 설정에도 반영 (안전장치)
        model.config.encoder.image_size = [1280, 960]
        
        model.to(device)
        model.eval()
        print(f"✅ AI 모델 로딩 완료!")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        model = None
        raise e

def run_inference(image_input):
    if model is None:
        try:
            load_model_lazy()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    try:
        if image_input.mode != "RGB":
            image_input = image_input.convert("RGB")

        # 전처리
        pixel_values = processor(image_input, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        task_prompt = "<s_receipt>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)

        # 생성
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1, # 속도 위해 1로 설정 (필요 시 4)
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        print(f"🤖 AI 분석 결과: {sequence}")

        try:
            json_output = processor.token2json(sequence)
            return {"status": "success", "result": json_output}
        except Exception as json_err:
            return {"status": "partial_success", "result": {"text_content": sequence}}

    except Exception as e:
        return {"status": "error", "message": str(e)}