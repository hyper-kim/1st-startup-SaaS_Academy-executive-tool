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
    scan_text_for_students,
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
        print(f"'{image.name}' 이미지 OCR 처리 시작...")
        ocr_text = call_clova_ocr_api(image)
        
        if "ERROR:" in ocr_text:
            return [f"OCR 처리 실패: {ocr_text}"]
        
        print(f"OCR Raw Text:\n{ocr_text}") # 디버깅용 출력
        
        # 텍스트 처리 로직으로 넘김
        return self._process_text_data(ocr_text)

    def _process_text_data(self, text):
        """
        [전략 변경]
        1. 텍스트 전체에서 'DB에 있는 학생 이름'을 먼저 싹 찾습니다. (가장 정확)
        2. 이름이 발견되면 -> '이름 매칭 성공'으로 처리.
        3. 이름이 없으면 -> 텍스트의 모든 줄에서 '숫자(금액)'를 찾아 1:1 매칭 시도.
        """
        results = []
        
        # --- 전략 1: 이름 기반 검색 (Priority 1) ---
        found_students = scan_text_for_students(text)
        
        if found_students:
            for student in found_students:
                # (심화: 여기서 해당 학생 이름 근처의 금액을 찾는 로직을 추가할 수도 있음)
                results.append(f"✅ 이름 매칭 성공: '{student.name}' 학생 발견! (수강료: {student.base_fee}원)")
            
            # 이름을 찾았더라도, 이름 없는 영수증이 섞여있을 수 있으니 
            # 금액 검색도 계속 진행할지 여부는 선택 사항입니다. 
            # 일단 여기서는 이름 찾으면 return 하지 않고 아래 금액 로직도 돌려보겠습니다.
            # (만약 중복이 싫으면 여기서 return results 하세요)
        
        
        # --- 전략 2: 금액 기반 검색 (Priority 2) ---
        # 텍스트를 한 줄씩 읽으며 금액 패턴을 찾습니다.
        lines = text.splitlines()
        for line in lines:
            # 라벨(성명 등) 제거 등 복잡한 정규식 다 버리고, 오직 '숫자'만 봅니다.
            
            # 1. "8 만원" 패턴 (수기 영수증)
            match_manwon = re.search(r'([\d,]+)\s*만원', line)
            
            # 2. "250,000" 패턴 (일반)
            #    (전화번호, 날짜 등 오인식 방지를 위해 1000원 이상, 콤마 포함 등을 조건으로 검)
            match_amount = re.search(r'([\d]{1,3}(?:,[\d]{3})+)', line) # 250,000 처럼 콤마가 있는 숫자
            
            amount = 0
            
            if match_manwon:
                # "8" -> 80000
                num_str = match_manwon.group(1).replace(',', '')
                amount = int(num_str) * 10000
                
            elif match_amount:
                # "250,000" -> 250000
                num_str = match_amount.group(1).replace(',', '')
                amount = int(num_str)
            
            # 숫자가 너무 작거나(날짜), 너무 크면(전화번호) 무시
            if amount < 1000 or amount > 5000000:
                continue

            # 이미 이름으로 찾은 학생 중에 이 금액을 가진 학생이 있다면 중복 처리 방지
            is_already_found = False
            for s in found_students:
                # (수강료 또는 교재비와 일치하면 스킵)
                if s.base_fee == amount or s.book_fee == amount:
                    is_already_found = True
                    break
            
            if is_already_found:
                continue

            # DB 매칭 시도 (1:1)
            student = find_student_by_amount(amount)
            if student:
                results.append(f"💰 금액 매칭 성공: {amount}원 -> {student.name}")
            else:
                # 1:1 실패 시 합산 매칭(N:1) 시도
                matches = find_payment_matches(amount)
                if matches['type'] == 'N:1':
                    names = ", ".join([s.name for s in matches['students']])
                    results.append(f"💡 합산 제안: {amount}원 -> {names} 합산?")
                
                # 실패 로그는 너무 많이 뜨면 지저분하므로, 확실한 금액 패턴일 때만 출력
                # results.append(f"❓ 매칭 실패: {amount}원 (학생 못 찾음)")

        if not results:
            results.append("❌ 매칭 실패: 인식된 이름이나 매칭되는 금액이 없습니다.")

        return results