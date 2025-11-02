from fuzzywuzzy import fuzz
from .models import Student

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