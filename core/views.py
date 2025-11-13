# core/views.py

import re
import json
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Student, Payment
from .serializers import StudentSerializer, PaymentSerializer

# AI 로직 및 OCR API 호출을 위해 services.py에서 함수들을 가져옵니다.
from .services import (
    find_student_by_amount, 
    find_student_by_name, 
    call_clova_ocr_api,
    find_payment_matches # (F-AI-02: 합산 결제)
)

# -----------------------------------------------------------------
# 1. 학생 관리 ViewSet (CRUD + 텍스트 일괄 등록)
# -----------------------------------------------------------------
class StudentViewSet(viewsets.ModelViewSet):
    """
    학생 정보를 관리(CRUD)하고,
    텍스트 블록으로 학생 명단을 일괄 등록합니다.
    """
    queryset = Student.objects.all().order_by('name')
    serializer_class = StudentSerializer

    @action(detail=False, methods=['post'])
    def upload_text_batch(self, request):
        """
        /api/students/upload_text_batch/
        
        원장님이 복사/붙여넣기 한 텍스트 블록을 파싱하여
        학생들을 일괄 등록합니다.
        
        [입력 형식 (JSON)]
        { "student_data": "노*연 250000\n이*창 250000\n박*재 80000" }
        """
        raw_text = request.data.get('student_data')
        # 1. ❗️ 이 코드가 'NoneType' 오류를 해결합니다.
        #    raw_text가 None이거나 빈 문자열("")이면 여기서 중단됩니다.
        if not raw_text:
            return Response({"error": "텍스트 데이터가 없습니다."}, status=status.HTTP_400_BAD_REQUEST)

        students_to_create = []
        
        try:
            # 2. 이 코드가 실행될 땐 raw_text는 절대 None이 아닙니다.
            lines = raw_text.strip().splitlines() 
            
            for line in lines:
                if not line.strip(): # 빈 줄 건너뛰기
                    continue
                
                # 3. ❗️ 이 정규 표현식이 "이름 수강료" 및 "이름 수강료 교재비 금액" 형식을 모두 처리합니다.
                match = re.search(
                    r'^\s*([^\d\s]+[\w*\s]*)\s+([\d,]+)\s*(?:교재비\s+([\d,]+))?\s*(.*)$', 
                    line
                )
                
                if not match:
                    # (예: "노*연 250000" 형식에 맞지 않는 줄)
                    raise ValueError(f"'{line}' 줄의 형식이 올바르지 않습니다.")

                name = match.group(1).strip()
                base_fee_str = re.sub(r',', '', match.group(2))
                
                # "교재비" 그룹이 인식되면(match.group(3)) 숫자로 변환, 없으면 0
                book_fee_str = re.sub(r',', '', match.group(3)) if match.group(3) else '0'
                
                notes = match.group(4).strip() if match.group(4) else '' # 4번 그룹도 None이 아님
                
                student = Student(
                    name=name,
                    base_fee=int(base_fee_str),
                    book_fee=int(book_fee_str), # 교재비 저장
                    notes=notes
                )
                students_to_create.append(student)

            # bulk_create로 DB에 한 번에 저장
            Student.objects.bulk_create(students_to_create)

            return Response(
                {"status": "success", "count": len(students_to_create)}, 
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            return Response(
                {"error": f"파일 처리 중 오류 발생: {e}"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

# -----------------------------------------------------------------
# 2. 결제 내역 관리 ViewSet (기본 CRUD)
# -----------------------------------------------------------------
class PaymentViewSet(viewsets.ModelViewSet):
    """
    개별 결제 내역을 관리(CRUD)합니다.
    (주로 AI가 생성하지만, 수동으로 수정/삭제할 때 사용됩니다.)
    """
    queryset = Payment.objects.all().order_by('-payment_date')
    serializer_class = PaymentSerializer

# -----------------------------------------------------------------
# 3. AI 정산 매칭 ViewSet (핵심 기능)
# -----------------------------------------------------------------
class MatchingViewSet(viewsets.ViewSet):
    """
    /api/matching/upload_data/
    
    원장님이 던져주는 모든 데이터(텍스트, 이미지)를 받아
    AI 매칭 로직을 실행합니다.
    """
    
    @action(detail=False, methods=['post'])
    def upload_data(self, request):
        
        text_data = request.data.get('text_input')
        image_file = request.FILES.get('image_file')

        if not text_data and not image_file:
            return Response({"error": "데이터가 없습니다. (텍스트 또는 이미지)"}, status=status.HTTP_400_BAD_REQUEST)
        
        matched_results = []
        
        # 1. 텍스트 입력이 있을 경우 (은행 이체 내역 등)
        if text_data:
            results = self._process_text_data(text_data)
            matched_results.extend(results)

        # 2. 이미지 입력이 있을 경우 (전표 사진, 수기 영수증)
        if image_file:
            results = self._process_image_data(image_file)
            matched_results.extend(results)
            
        return Response({
            "message": "자동 매칭 완료",
            "results": matched_results # 매칭 결과(처리 내역) 반환
        })

    # --- 내부 헬퍼(Helper) 함수들 ---
    
    def _process_image_data(self, image):
        """
        (Helper) 이미지를 OCR API로 전송하고, 반환된 텍스트를
        _process_text_data 함수로 넘겨 처리합니다.
        """
        
        print(f"'{image.name}' 이미지 OCR 처리 시작...")
        ocr_text = call_clova_ocr_api(image)
        
        if "ERROR:" in ocr_text:
            print(f"OCR 실패: {ocr_text}")
            return [f"OCR 처리 실패: {ocr_text}"]
        
        print(f"OCR 인식 결과:\n{ocr_text}")
        
        # OCR로 인식된 텍스트를 다시 텍스트 처리 로직으로 넘깁니다.
        return self._process_text_data(ocr_text)
    
    def _process_text_data(self, text):
        """
        (Helper) OCR 텍스트 또는 입력 텍스트를 파싱하여
        이름과 금액을 추출하고, AI 매칭을 시도합니다.
        """
        processed = []
        
        # '성 명' 또는 '성명' 라벨을 찾아 이름을 추출
        name_match = re.search(r'성\s?명\s*[:\s]\s*([^\n]+)', text, re.IGNORECASE)
        name = name_match.group(1).strip() if name_match else None
        
        amount = 0
        amount_str = ""

        # 1. "만원" 패턴 먼저 검색 (수기 영수증)
        amount_match_manwon = re.search(r'(?:교습비|납부 명세)\s*[:\s]*([\d,]+)\s*만원', text, re.IGNORECASE)
        
        # 2. "일반" 패턴 검색 (카드 전표, 은행 이체)
        amount_match_default = re.search(r'(?:합계\s?금액|입금|원주정산)\s*[:\s]*([\d,]+)', text, re.IGNORECASE)

        try:
            if amount_match_manwon:
                # Case 1: "만원" 패턴 매칭 성공 (예: 8 만원)
                amount_str = re.sub(r'[,\s]', '', amount_match_manwon.group(1)) # "8"
                amount = int(amount_str) * 10000 # 8 -> 80000
                    
            elif amount_match_default:
                # Case 2: "일반" 패턴 매칭 성공 (예: 250,000)
                amount_str = re.sub(r'[,\s원]', '', amount_match_default.group(1)) # "250000"
                amount = int(amount_str) # 250000
            
            else:
                # Case 3: 금액을 못 찾음 (텍스트가 여러 줄일 수 있으니 함수 종료 X)
                pass # 그냥 amount = 0
                
        except ValueError:
            processed.append(f"오류: '{amount_str}'을 숫자로 바꿀 수 없습니다.")
            amount = 0 # 금액 인식 실패 시 0으로 초기화
        

        # 3. 금액을 찾았으니, 이제 이름과 매칭 시도
        if name and amount > 0:
            # 이름과 금액이 모두 인식된 경우 (수기/카드 전표)
            student = find_student_by_name(name)
            if student:
                # (향후: Payment 객체 생성)
                processed.append(f"이름/금액 매칭 성공: {name} 학생, {amount}원")
            else:
                processed.append(f"매칭 실패: {name} 학생을 DB에서 찾을 수 없음")
        
        elif amount > 0:
            # 이름 없이 금액만 인식된 경우 (은행 이체)
            student = find_student_by_amount(amount) # 1:1 금액 매칭
            if student:
                processed.append(f"금액(1:1) 매칭 성공: {amount}원 -> {student.name}")
            else:
                # 1:1 실패 시 '합산 결제' 시도 (F-AI-02)
                matches = find_payment_matches(amount)
                if matches['type'] == 'N:1':
                    student_names = ", ".join([s.name for s in matches['students']])
                    processed.append(f"금액(N:1) 매칭 제안: {amount}원 -> {student_names} 학생들의 합산?")
                else:
                    processed.append(f"매칭 실패: {amount}원 (1:1 및 N:1 매칭 모두 실패)")
        
        else:
            processed.append(f"매칭 실패: 텍스트에서 유효한 '금액' 패턴을 찾을 수 없음")

        return processed