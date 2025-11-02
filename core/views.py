from django.shortcuts import render

# Create your views here.
import re  # 텍스트에서 숫자(금액)를 찾기 위한 '정규 표현식' 라이브러리
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Student, Payment
from .serializers import StudentSerializer, PaymentSerializer

# 💡 1. 엑셀 대신 '금액'으로 학생을 찾는 새 AI 로직을 가져옵니다.
from .services import find_student_by_amount

# 'ModelViewSet'은 API의 모든 기본 동작(CRUD)을 자동으로 만들어줍니다.
# (CRUD: Create, Retrieve, Update, Delete)

class StudentViewSet(viewsets.ModelViewSet):
    queryset = Student.objects.all() # 이 API가 다룰 데이터(모든 학생)
    serializer_class = StudentSerializer # 1단계에서 만든 번역기 지정

class PaymentViewSet(viewsets.ModelViewSet):
    queryset = Payment.objects.all()
    serializer_class = PaymentSerializer

class MatchingViewSet(viewsets.ViewSet):

    # /api/matching/upload_excel/ 주소로 POST 요청을 받습니다.
    @action(detail=False, methods=['post'])
    def upload_data(self, request):
        
        text_data = request.data.get('text_input')
    
        # 2. 이미지 파일이 있는지 확인
        image_file = request.FILES.get('image_file')

        if not text_data and not image_file:
            return Response({"error": "데이터가 없습니다. (텍스트 또는 이미지)"}, status=400)

        matched_results = []
        
        # 4. 텍스트 입력이 있을 경우 (이모님 카톡 메모)
        if text_data:
            # 텍스트를 파싱(분석)해서 매칭을 시도합니다.
            results = self._process_text_data(text_data)
            matched_results.extend(results)

        # 5. 이미지 입력이 있을 경우 (전표 사진, 은행 앱 스크린샷)
        if image_file:
            # 이미지를 OCR로 분석해서 매칭을 시도합니다.
            results = self._process_image_data(image_file)
            matched_results.extend(results)
            
        return Response({
            "message": "자동 매칭 완료",
            "results": matched_results # 매칭 결과 리스트 반환
        })

    # --- 내부 헬퍼(Helper) 함수들 ---
    
    def _process_text_data(self, text):
        """
        입력된 텍스트(여러 줄)를 분석해 금액을 찾고 학생과 매칭합니다.
        (예: "제로페이결제사 60,000원")
        """
        processed = []
        
        # 텍스트를 한 줄씩 분석합니다.
        for line in text.splitlines():
            # 're.sub'를 사용해 쉼표(,)와 '원' 글자를 제거하고 숫자만 찾습니다.
            cleaned_line = re.sub(r'[,\s원]', '', line)
            
            # 숫자(금액)를 찾습니다.
            match = re.search(r'(\d+)', cleaned_line)
            if not match:
                continue # 이 줄에 금액이 없으면 통과
            
            amount = int(match.group(1)) # (예: 60000)
            
            # 💡 6. 새 AI 로직(금액 기반) 호출!
            student = find_student_by_amount(amount)
            
            if student:
                # (여기서 Payment 객체를 생성/업데이트하면 됩니다)
                processed.append(f"성공: {line} -> {student.name} 학생 (기준금액: {student.base_fee}원)")
            else:
                processed.append(f"실패: {line} (금액: {amount}원) -> 일치하는 학생 없음")
        
        return processed

    def _process_image_data(self, image):
        """
        입력된 이미지를 OCR API로 전송하고, 반환된 텍스트를
        다시 _process_text_data 함수로 넘겨 처리합니다.
        """
        
        # 1. (필수 구현) 네이버 CLOVA OCR API 호출 로직
        #    image 파일을 OCR API로 전송합니다.
        # ocr_text = call_naver_ocr_api(image) 
        
        # --- (아래는 OCR API를 구현했다는 가정 하의 테스트용 코드) ---
        # (실제 구현 시 위 OCR API 호출 코드로 대체해야 합니다)
        print(f"'{image.name}' 이미지 처리 시뮬레이션 (OCR API 호출 필요)")
        ocr_text = "NH16140241 249,625원\nKB10410261 149,400원" # OCR 결과 텍스트 (가상)
        # --- (테스트용 코드 끝) ---
        
        # 2. OCR로 인식된 텍스트를 다시 텍스트 처리 함수로 넘깁니다.
        return self._process_text_data(ocr_text)