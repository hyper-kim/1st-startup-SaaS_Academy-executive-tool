# core/services.py

import requests
import json
import uuid
import time
import re
from itertools import combinations
from django.conf import settings
from fuzzywuzzy import fuzz
from .models import Student, Payment

# -----------------------------------------------------------------
# 1. Naver CLOVA OCR API Service
# -----------------------------------------------------------------
def call_clova_ocr_api(image_file):
    """
    이미지 파일(jpg, png)을 받아 네이버 CLOVA OCR API를 호출하고,
    인식된 텍스트(여러 줄)를 반환합니다.
    """
    
    api_url = settings.CLOVA_API_URL
    secret_key = settings.CLOVA_SECRET_KEY
    
    if not api_url or not secret_key:
        print("ERROR: OCR API Key가 .env 또는 settings.py에 설정되지 않았습니다.")
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

    try:
        response = requests.post(api_url, headers=headers, data=payload, files=files)
        response.raise_for_status() # 4xx, 5xx 에러 시 예외 발생
        
        result = response.json()
        
        # OCR 결과에서 텍스트만 추출하여 하나의 문자열로 합칩니다.
        full_text = ""
        for field in result['images'][0]['fields']:
            full_text += field['inferText'] + "\n" # 각 텍스트 조각을 줄바꿈으로 연결
            
        return full_text.strip()
    
    except requests.RequestException as e:
        print(f"OCR API Error: {e}")
        return f"ERROR: OCR API 호출 실패 - {e}"

# -----------------------------------------------------------------
# 2. AI Matching Service (Name-based)
# -----------------------------------------------------------------
def find_student_by_name(ocr_name):
    """
    OCR로 인식된 이름(예: '박*재', '노*연(중등수학)')을 받아서,
    DB의 학생 이름과 비교해 가장 일치하는 학생을 찾습니다.
    """
    # 이름에서 "(중등수학)" 같은 괄호 안 메모를 제거
    cleaned_name = re.sub(r'\(.*\)', '', ocr_name).strip()
    
    students = Student.objects.all() # (향후 '미납' 학생만 필터링)
    best_score = 0
    best_match = None
    
    for student in students:
        # '노*연(중등수학)'과 '노*연'을 비교하기 위해 partial_ratio 사용
        score = fuzz.partial_ratio(cleaned_name.lower(), student.name.lower())
        
        if score > best_score:
            best_score = score
            best_match = student
            
    # 85점 이상일 때만 동일인으로 간주 (오인식 방지)
    if best_score >= 85:
        return best_match
    else:
        return None

# -----------------------------------------------------------------
# 3. AI Matching Service (Amount-based, 1:1)
# -----------------------------------------------------------------
def find_student_by_amount(paid_amount, tolerance=1000):
    """
    (1:1 매칭) 입금액을 '수강료' 또는 '교재비'와 비교하여
    일치하는 '미납' 학생 1명을 찾습니다.
    """
    
    min_fee = paid_amount - tolerance
    max_fee = paid_amount + tolerance
    
    from django.db.models import Q # 👈 Q 객체 임포트 (OR 조건용)

    # 👇 [수정] DB 조회 로직
    # "base_fee가 범위 내에 '또는(OR)' book_fee가 범위 내에 있는 학생"
    possible_matches = Student.objects.filter(
        Q(base_fee__range=(min_fee, max_fee)) |
        Q(book_fee__range=(min_fee, max_fee))
        # (향후: Q(base_fee + book_fee ... ) 합산 로직도 추가 가능)
    )
    
    if possible_matches.count() == 1:
        return possible_matches.first()
    else:
        # (만약 32,000원이 교재비인 학생이 여러 명이라 헷갈리면 실패 처리)
        return None

# -----------------------------------------------------------------
# 4. AI Matching Service (Amount-based, N:1 - Killer Feature)
# -----------------------------------------------------------------
def find_payment_matches(paid_amount, tolerance=1000, max_batch_size=3):
    """
    (N:1 매칭) 입금액을 받아, 1:1 매칭 실패 시 
    '미납' 학생들의 수강료 '조합'으로 합산 매칭을 시도합니다.
    """
    
    # (향후: .filter(status='UNPAID') 등을 추가)
    unpaid_students = list(Student.objects.all())
    
    # --- 1. (1:1 매칭) 단일 학생 매칭 시도 ---
    # (find_student_by_amount 함수 로직을 여기서 먼저 수행)
    
    possible_matches_1_to_1 = []
    for student in unpaid_students:
        if abs(student.base_fee - paid_amount) <= tolerance:
            possible_matches_1_to_1.append(student)

    if len(possible_matches_1_to_1) == 1:
        return {'type': '1:1', 'students': possible_matches_1_to_1}

    # --- 2. (N:1 매칭) 합산 결제 매칭 시도 ---
    
    # (성능을 위해 최대 3명까지의 조합만 확인)
    for batch_size in range(2, max_batch_size + 1):
        for student_batch in combinations(unpaid_students, batch_size):
            
            # (예: (박*재, 이*준) 학생 조합)
            total_fee = sum(student.base_fee for student in student_batch)
            
            # (예: 80,000 + 140,000 = 220,000)
            if abs(total_fee - paid_amount) <= tolerance:
                # 합산 매칭 성공!
                return {
                    'type': 'N:1',
                    'students': list(student_batch)
                }

    # 1:1, N:1 매칭 모두 실패
    return {'type': 'FAIL', 'students': []}