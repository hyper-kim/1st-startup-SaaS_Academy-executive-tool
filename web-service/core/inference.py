import torch
import re
import json
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# -----------------------------------------------------------------------------
# ★ [설정] 본인의 Hugging Face 모델 ID로 바꿔주세요
# 형식: "사용자아이디/모델명"
# 예시: "hyper-kim/saas-receipt-model"
# -----------------------------------------------------------------------------
MODEL_ID = "HYPER-KJY/academy-receipt-model"

# 전역 변수
model = None
processor = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_lazy():
    """
    최초 요청 시 Hugging Face Hub에서 모델을 다운로드/로드합니다.
    """
    global model, processor
    
    if model is not None:
        return

    print(f"💤 Hugging Face Hub에서 모델을 찾아오는 중... (ID: {MODEL_ID})")

    try:
        # ---------------------------------------------------------
        # Hugging Face Hub 자동 로드 (인터넷 연결 필수)
        # ---------------------------------------------------------
        # 만약 비공개(Private) 모델이라면, 터미널에서 'huggingface-cli login'을 했거나
        # token="hf_..." 인자를 추가해야 합니다.
        
        # 1. 프로세서 로드
        try:
            processor = DonutProcessor.from_pretrained(MODEL_ID)
        except OSError:
            # 혹시나 설정 파일이 꼬였을 경우를 대비한 안전장치
            print("⚠️ 모델 저장소에 프로세서 설정이 없어 기본값(donut-base)을 사용합니다.")
            processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            processor.tokenizer.add_tokens(["<s_receipt>", "</s_receipt>"])

        # 2. 모델 로드
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
        
        # 토큰 크기 맞춤
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        model.to(device)
        model.eval()
        print(f"✅ AI 모델 로딩 완료! (Source: Hugging Face Hub)")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        model = None
        raise e

def run_inference(image_input):
    """
    views.py에서 호출하는 추론 함수
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

        # 4. 생성 (Inference) - 품질 옵션 적용
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=768,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                # 앵무새 방지 옵션
                num_beams=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # 5. 후처리
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        print(f"🤖 AI 분석 결과(Raw): {sequence}")

        # 6. JSON 파싱
        try:
            json_output = processor.token2json(sequence)
            return {"status": "success", "result": json_output}
        except Exception as json_err:
            return {"status": "partial_success", "result": {"text_content": sequence}}

    except Exception as e:
        return {"status": "error", "message": str(e)}