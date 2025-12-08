import json
import pandas as pd
import random
from faker import Faker

# 한국어 설정
fake = Faker('ko_KR')
NUM_STUDENTS = 5500 # 넉넉하게 5500개 생성 (텀프 기준 5000개 충족)

# 학원 데이터베이스
COURSES = [
    {"name": "초등 사고력 수학", "fee": 180000, "category": "Math"},
    {"name": "중등 내신 수학", "fee": 250000, "category": "Math"},
    {"name": "고등 수능 수학", "fee": 350000, "category": "Math"},
    {"name": "초등 파닉스 영어", "fee": 200000, "category": "English"},
    {"name": "중등 문법/독해", "fee": 280000, "category": "English"},
    {"name": "고등 수능 영어", "fee": 320000, "category": "English"},
    {"name": "입시 논술", "fee": 400000, "category": "Essay"}
]

def generate_student_db():
    students = []
    print(f"🔥 학생 데이터 {NUM_STUDENTS}명 생성 시작...")

    for i in range(NUM_STUDENTS):
        course = random.choice(COURSES)
        
        # 텀프로젝트용 Feature 확장 (10개 이상)
        student = {
            "student_id": f"STU{i:05d}",               # 1. ID
            "name": fake.name(),                        # 2. 이름
            "gender": random.choice(["M", "F"]),        # 3. 성별
            "age": random.randint(8, 19),               # 4. 나이
            "address_city": fake.city(),                # 5. 거주지(시)
            "phone": fake.phone_number(),               # 6. 전화번호
            "parent_name": fake.name(),                 # 7. 학부모 성명
            "registration_date": fake.date_this_decade().isoformat(), # 8. 등록일
            "course_name": course["name"],              # 9. 수강 과목
            "category": course["category"],             # 10. 과목 카테고리
            "base_fee": course["fee"],                  # 11. 수강료
            "book_fee": random.choice([0, 20000, 30000, 50000]), # 12. 교재비
            "payment_method": random.choice(["Card", "Cash", "Transfer"]), # 13. 결제수단
            "is_dropout": random.choices([0, 1], weights=[0.8, 0.2])[0] # 14. 이탈여부 (Target)
        }
        students.append(student)

    # 1. 시스템용 JSON 저장
    with open("mock_data/student_db.json", "w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)
    
    # 2. 텀프로젝트/분석용 CSV 저장
    df = pd.DataFrame(students)
    df.to_csv("mock_data/student_list.csv", index=False, encoding="utf-8-sig")
    
    print(f"✅ 생성 완료! (JSON: {len(students)}개, CSV: {len(df)}행)")
    print("👉 mock_data/student_list.csv 파일을 텀프로젝트 분석에 사용하세요.")

if __name__ == "__main__":
    import os
    os.makedirs("mock_data", exist_ok=True)
    generate_student_db()