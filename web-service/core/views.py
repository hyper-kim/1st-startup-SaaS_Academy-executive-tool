# web-service/core/views.py

import re
import json
import io
from PIL import Image

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Student, Payment
from .serializers import StudentSerializer, PaymentSerializer

# 로컬 AI 엔진 가져오기
from .inference import run_inference

# 기존 서비스 로직 (DB 매칭용)
from .services import (
    find_student_by_amount, 
    scan_text_for_students,
    find_payment_matches
)

# -----------------------------------------------------------------
# 1. 학생 관리 ViewSet
# -----------------------------------------------------------------
class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all().order_by('name')
    serializer_class = StudentSerializer

    @action(detail=False, methods=['post'])
    def upload_text_batch(self, request):
        """ 학생 명단 텍스트 일괄 등록 """
        raw_text = request.data.get('student_data')
        if not raw_text:
            return Response({"error": "데이터가 없습니다."}, status=status.HTTP_400_BAD_REQUEST)

        students_to_create = []
        try:
            lines = raw_text.strip().splitlines()
            for line in lines:
                if not line.strip(): continue
                
                # 정규식: 이름 금액 [교재비 금액] 비고
                match = re.search(r'^\s*([^\d\s]+[\w*\s]*)\s+([\d,]+)\s*(?:교재비\s+([\d,]+))?\s*(.*)$', line)
                if not match: continue

                name = match.group(1).strip()
                base_fee = int(re.sub(r',', '', match.group(2)))
                book_fee = int(re.sub(r',', '', match.group(3))) if match.group(3) else 0
                notes = match.group(4).strip() if match.group(4) else ''
                
                students_to_create.append(Student(
                    name=name, base_fee=base_fee, book_fee=book_fee, notes=notes
                ))

            Student.objects.bulk_create(students_to_create)
            return Response({"status": "success", "count": len(students_to_create)}, status=status.HTTP_201_CREATED)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

# -----------------------------------------------------------------
# 2. 결제 내역 관리 ViewSet
# -----------------------------------------------------------------
class PaymentViewSet(viewsets.ModelViewSet):
    queryset = Payment.objects.all().order_by('-payment_date')
    serializer_class = PaymentSerializer

# -----------------------------------------------------------------
# 3. AI 정산 매칭 ViewSet (핵심 기능)
# -----------------------------------------------------------------
class MatchingViewSet(viewsets.ViewSet):
    
    @action(detail=False, methods=['post'])
    def upload_data(self, request):
        text_data = request.data.get('text_input')
        image_file = request.FILES.get('image_file')

        if not text_data and not image_file:
            return Response({"error": "텍스트 또는 이미지를 입력해주세요."}, status=status.HTTP_400_BAD_REQUEST)
        
        matched_results = []
        
        # 1. 텍스트 직접 입력 처리
        if text_data:
            results = self._process_text_data(text_data)
            matched_results.extend(results)

        # 2. 이미지 파일 처리 (AI 모델 추론)
        if image_file:
            image_results = self._process_image_data(image_file)
            matched_results.extend(image_results)
            
        return Response({
            "message": "분석 완료",
            "results": matched_results 
        })

    def _process_image_data(self, image_file):
        """ 이미지를 AI 모델에 넣어 JSON 결과를 받고, 텍스트로 변환하여 분석 """
        try:
            # 1. 이미지 변환 (Django UploadFile -> PIL)
            image_bytes = image_file.read()
            pil_image = Image.open(io.BytesIO(image_bytes))

            # 2. AI 추론 실행
            ai_output = run_inference(pil_image)

            if ai_output['status'] != 'success':
                # 부분 성공으로 텍스트만 왔을 경우도 처리 가능하지만, 여기선 에러 처리
                if ai_output.get('status') == 'partial_success':
                     return self._process_text_data(ai_output['result'].get('text_content', ''))
                return [f"❌ AI 분석 실패: {ai_output.get('message')}"]

            # 3. JSON 데이터 추출
            data = ai_output['result']
            print(f"🔍 AI 추출 JSON 데이터: {data}")

            # 4. [중요] JSON 데이터를 기존 텍스트 분석 로직이 이해할 수 있는 '문자열'로 변환
            # 예: {'total_price': '50,000', 'student': '홍길동'} -> "학생: 홍길동\n금액: 50,000"
            converted_lines = []
            
            # (1) 학생 이름 추출
            if 'student' in data:
                converted_lines.append(f"학생명: {data['student']}")
            
            # (2) 총 금액 추출 (total_price 또는 amount 키)
            if 'total_price' in data:
                converted_lines.append(f"총계 {data['total_price']}")
            elif 'amount' in data:
                converted_lines.append(f"금액 {data['amount']}")
            
            # (3) 품목 내역 추출 (items 리스트)
            if 'items' in data and isinstance(data['items'], list):
                for item in data['items']:
                    # item이 dict인 경우 desc와 price 추출
                    if isinstance(item, dict):
                        desc = item.get('desc', item.get('item', ''))
                        price = item.get('price', item.get('amount', ''))
                        converted_lines.append(f"{desc} {price}")
                    elif isinstance(item, str):
                        converted_lines.append(item)

            full_text_from_ai = "\n".join(converted_lines)
            print(f"📝 변환된 분석 텍스트:\n{full_text_from_ai}")

            # 5. 변환된 텍스트로 매칭 로직 실행
            return self._process_text_data(full_text_from_ai)

        except Exception as e:
            print(f"Image Processing Error: {e}")
            return [f"서버 에러: 이미지 처리 중 문제가 발생했습니다. {str(e)}"]

    def _process_text_data(self, text):
        """ 텍스트에서 학생 이름과 금액을 찾아 DB와 매칭 """
        results = []
        
        # 1. 이름 기반 검색
        found_students = scan_text_for_students(text)
        if found_students:
            for student in found_students:
                results.append(f"✅ 이름 매칭: '{student.name}' 학생 (DB 수강료: {student.base_fee:,}원)")

        # 2. 금액 기반 검색
        lines = text.splitlines()
        for line in lines:
            # 숫자만 추출 (콤마 제거)
            numbers = re.findall(r'\d+', line.replace(',', ''))
            
            for num_str in numbers:
                amount = int(num_str)
                
                # 금액 노이즈 필터링
                if amount < 1000 or amount > 10000000:
                    continue

                # 이미 찾은 학생의 수강료와 같다면 중복 출력 방지
                is_already_found = False
                for s in found_students:
                    if s.base_fee == amount or s.book_fee == amount:
                        is_already_found = True
                        break
                if is_already_found:
                    continue

                # 금액 매칭 시도
                student = find_student_by_amount(amount)
                if student:
                    results.append(f"💰 금액 매칭: {amount:,}원 → {student.name}")
                else:
                    # 합산 매칭 시도
                    matches = find_payment_matches(amount)
                    if matches['type'] == 'N:1':
                        names = ", ".join([s.name for s in matches['students']])
                        results.append(f"💡 합산 의심: {amount:,}원 → {names} 합산액과 일치")

        if not results:
            results.append("❌ 매칭 실패: 텍스트에서 유의미한 정보를 찾지 못했습니다.")

        return results