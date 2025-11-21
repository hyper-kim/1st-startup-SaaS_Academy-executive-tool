# core/views.py

import re
import json
import io
from PIL import Image  # 이미지 처리를 위해 추가

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Student, Payment
from .serializers import StudentSerializer, PaymentSerializer

# 1. 우리가 만든 로컬 AI 엔진 가져오기
from .inference import run_inference 

# 2. 기존 서비스 로직 (DB 매칭용)은 그대로 사용
from .services import (
    find_student_by_amount, 
    scan_text_for_students,
    find_student_by_name, 
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
        원장님이 복사/붙여넣기 한 텍스트 블록을 파싱하여 학생들을 일괄 등록합니다.
        """
        raw_text = request.data.get('student_data')
        if not raw_text:
            return Response({"error": "텍스트 데이터가 없습니다."}, status=status.HTTP_400_BAD_REQUEST)

        students_to_create = []
        
        try:
            lines = raw_text.strip().splitlines() 
            
            for line in lines:
                if not line.strip(): continue
                
                # 이름, 수강료, (선택)교재비, (선택)비고 파싱
                match = re.search(
                    r'^\s*([^\d\s]+[\w*\s]*)\s+([\d,]+)\s*(?:교재비\s+([\d,]+))?\s*(.*)$', 
                    line
                )
                
                if not match:
                    # 형식이 안 맞으면 일단 에러 로그만 남기고 패스하거나 에러 리턴 (여기선 에러 리턴)
                    raise ValueError(f"'{line}' 줄의 형식이 올바르지 않습니다.")

                name = match.group(1).strip()
                base_fee = int(re.sub(r',', '', match.group(2)))
                book_fee = int(re.sub(r',', '', match.group(3))) if match.group(3) else 0
                notes = match.group(4).strip() if match.group(4) else ''
                
                student = Student(
                    name=name,
                    base_fee=base_fee,
                    book_fee=book_fee,
                    notes=notes
                )
                students_to_create.append(student)

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
# 2. 결제 내역 관리 ViewSet
# -----------------------------------------------------------------
class PaymentViewSet(viewsets.ModelViewSet):
    queryset = Payment.objects.all().order_by('-payment_date')
    serializer_class = PaymentSerializer

# -----------------------------------------------------------------
# 3. AI 정산 매칭 ViewSet (핵심 기능 수정됨)
# -----------------------------------------------------------------
class MatchingViewSet(viewsets.ViewSet):
    """
    원장님이 던져주는 데이터(텍스트, 이미지)를 받아
    Local AI 모델(inference.py)을 통해 분석하고 매칭합니다.
    """
    
    @action(detail=False, methods=['post'])
    def upload_data(self, request):
        
        text_data = request.data.get('text_input')
        image_file = request.FILES.get('image_file')

        if not text_data and not image_file:
            return Response({"error": "데이터가 없습니다. (텍스트 또는 이미지)"}, status=status.HTTP_400_BAD_REQUEST)
        
        matched_results = []
        
        # 1. 텍스트 입력이 있을 경우 (은행 복붙 등)
        if text_data:
            results = self._process_text_data(text_data)
            matched_results.extend(results)

        # 2. 이미지 입력이 있을 경우 (영수증 사진 -> AI 추론)
        if image_file:
            image_results = self._process_image_data(image_file)
            matched_results.extend(image_results)
            
        return Response({
            "message": "AI 자동 분석 및 매칭 완료",
            "results": matched_results 
        })

    # --- 내부 헬퍼 함수 ---
    
    def _process_image_data(self, image_file):
        """
        업로드된 이미지를 PIL로 변환하여 Local AI 모델(run_inference)에 전달합니다.
        """
        try:
            print(f"📸 '{image_file.name}' 이미지 AI 분석 시작...")
            
            # 1. Django UploadedFile -> Bytes -> PIL Image 변환
            image_bytes = image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

            # 2. Local AI 추론 실행 (inference.py)
            ai_response = run_inference(pil_image)

            # 3. 결과 확인
            if ai_response['status'] == 'success':
                extracted_text = ai_response['result']
                print(f"🤖 AI 추출 텍스트:\n{extracted_text}")
                
                # 추출된 텍스트를 기존 텍스트 분석 로직에 태움
                return self._process_text_data(extracted_text)
            else:
                return [f"❌ AI 분석 에러: {ai_response.get('message', '알 수 없는 오류')}"]

        except Exception as e:
            return [f"❌ 이미지 처리 중 서버 오류: {str(e)}"]

    def _process_text_data(self, text):
        """
        텍스트에서 학생 이름과 금액을 찾아 DB와 매칭합니다.
        (AI가 이미지를 텍스트로 바꿔주면, 이 함수가 분석을 담당합니다)
        """
        results = []
        
        # --- 전략 1: 이름 기반 검색 ---
        found_students = scan_text_for_students(text)
        
        if found_students:
            for student in found_students:
                results.append(f"✅ 이름 매칭 성공: '{student.name}' (수강료: {student.base_fee:,}원)")
        
        # --- 전략 2: 금액 기반 검색 ---
        lines = text.splitlines()
        for line in lines:
            # 1. "8 만원" 패턴
            match_manwon = re.search(r'([\d,]+)\s*만원', line)
            # 2. "250,000" 패턴
            match_amount = re.search(r'([\d]{1,3}(?:,[\d]{3})+)', line)
            
            amount = 0
            if match_manwon:
                amount = int(match_manwon.group(1).replace(',', '')) * 10000
            elif match_amount:
                amount = int(match_amount.group(1).replace(',', ''))
            
            # 노이즈 필터링 (1000원 미만, 500만원 초과 무시)
            if amount < 1000 or amount > 5000000:
                continue

            # 이미 이름으로 찾은 학생의 수강료라면 패스 (중복 방지)
            is_already_found = False
            for s in found_students:
                if s.base_fee == amount or s.book_fee == amount:
                    is_already_found = True
                    break
            if is_already_found:
                continue

            # DB 매칭 시도 (1:1)
            student = find_student_by_amount(amount)
            if student:
                results.append(f"💰 금액 매칭 성공: {amount:,}원 -> {student.name}")
            else:
                # 1:1 실패 시 합산 매칭(N:1) 시도
                matches = find_payment_matches(amount)
                if matches['type'] == 'N:1':
                    names = ", ".join([s.name for s in matches['students']])
                    results.append(f"💡 합산 제안: {amount:,}원 -> {names} 합산 가능성 있음")

        if not results:
            results.append("❌ 매칭 실패: 텍스트에서 유의미한 정보를 찾지 못했습니다.")

        return results