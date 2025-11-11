from fuzzywuzzy import fuzz
from .models import Student
import requests
import uuid
import time
from django.conf import settings
from .models import Student
import json

def find_student_by_amount(paid_amount, tolerance=1000):
    """
    입금된 금액 (예: 249,625원)을 받아서,
    DB의 학생 기본 수강료 (예: 250,000원)와
    오차 범위 (tolerance, 예: 1000원) 내에서 일치하는 학생을 찾습니다.
    
    (이 오차 범위는 나중에 수수료율을 계산해서 더 정교하게 만들 수 있습니다)
    """
    
    possible_matches = []
    
    # 1. 모든 학생의 (기본 수강료, 학생) 리스트를 가져옵니다.
    students = Student.objects.all() 
    
    for student in students:
        # 2. (학생 수강료 - 입금액)의 절댓값을 계산
        difference = abs(student.base_fee - paid_amount)
        
        # 3. 오차 범위(tolerance) 내에 있으면 후보로 추가
        if difference <= tolerance:
            possible_matches.append(student)
            
    # 4. 후보가 정확히 1명일 경우에만 매칭 성공으로 간주
    if len(possible_matches) == 1:
        return possible_matches[0]
    else:
        # 후보가 없거나, 2명 이상이라 헷갈리면(예: 25만원 학생 2명)
        # 매칭 실패로 처리 (원장이 수동 선택하도록 유도)
        return None
    
def call_clova_ocr_api(image_file):
    """
    이미지 파일(jpg, png)을 받아 네이버 CLOVA OCR API를 호출하고,
    인식된 텍스트(여러 줄)를 반환합니다.
    """
    
    # 1. settings.py에서 API 키 가져오기
    api_url = settings.CLOVA_API_URL
    secret_key = settings.CLOVA_SECRET_KEY
    
    if not api_url or not secret_key:
        return "ERROR: OCR API Key가 설정되지 않았습니다."

    request_json = {
        'images': [
            {
                'format': image_file.name.split('.')[-1], # jpg, png 등
                'name': 'temp_image'
            }
        ],
        'requestId': str(uuid.uuid4()),
        'version': 'V2',
        'timestamp': int(time.time() * 1000)
    }
    
    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', image_file.read())
    ]
    headers = {
        'X-OCR-SECRET': secret_key
    }

    # 2. 네이버 API에 요청 보내기
    try:
        response = requests.post(api_url, headers=headers, data=payload, files=files)
        response.raise_for_status() # 오류가 있으면 예외 발생
        
        result = response.json()
        
        # 3. OCR 결과에서 텍스트만 추출하기
        full_text = ""
        for field in result['images'][0]['fields']:
            full_text += field['inferText'] + "\n"
            
        return full_text.strip() # "금액: 250,000원\n이름: 노*연\n"
    
    except requests.RequestException as e:
        print(f"OCR API Error: {e}")
        return f"ERROR: OCR API 호출 실패 - {e}"