import json
import random
import csv
import os
from faker import Faker

fake = Faker('ko_KR')

# 설정
OUTPUT_DIR = "mock_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_student_db(count=100):
    students = []
    
    # 학원비/교재비 패턴 (현실성 부여)
    course_types = [
        {'name': '초등수학', 'fee': 150000, 'book': 15000},
        {'name': '중등수학', 'fee': 250000, 'book': 32000},
        {'name': '중등심화', 'fee': 300000, 'book': 35000},
        {'name': '고등수학', 'fee': 400000, 'book': 45000},
        {'name': '수리논술', 'fee': 500000, 'book': 0}, # 교재비 없는 경우
    ]

    for i in range(count):
        course = random.choice(course_types)
        
        student = {
            "id": i + 1,
            "name": fake.name(),
            "course_name": course['name'],
            "base_fee": course['fee'],
            "book_fee": course['book'],
            "parent_phone": fake.phone_number(),
            "memo": f"{course['name']}반" if random.random() > 0.5 else ""
        }
        students.append(student)

    # 1. JSON 저장 (영수증 생성기용)
    json_path = f"{OUTPUT_DIR}/student_db.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(students, f, ensure_ascii=False, indent=2)
        
    # 2. CSV 저장 (엑셀 확인용)
    csv_path = f"{OUTPUT_DIR}/student_list.csv"
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "이름", "과목", "수강료", "교재비", "연락처", "메모"])
        for s in students:
            writer.writerow([s['id'], s['name'], s['course_name'], s['base_fee'], s['book_fee'], s['parent_phone'], s['memo']])

    print(f"✅ 학생 명단 생성 완료!")
    print(f" - JSON: {json_path} (시스템용)")
    print(f" - CSV : {csv_path} (엑셀 확인용)")
    
    return students

if __name__ == "__main__":
    generate_student_db(100)