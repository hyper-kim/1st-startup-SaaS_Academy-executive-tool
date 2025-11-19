import os
import random
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import albumentations as A

# 설정
DATASET_DIR = "dataset/hardcore_train"
os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels", exist_ok=True)

fake = Faker('ko_KR')
FONT_PATH = "NanumGothic.ttf" 

# ---------------------------------------------------------
# 1. 복잡한 표 그리기 헬퍼 함수
# ---------------------------------------------------------
def draw_dashed_line(draw, y, width):
    # 점선 그리기 (----------------)
    draw.text((20, y), "-" * 45, font=ImageFont.truetype(FONT_PATH, 20), fill=(50, 50, 50))
    return y + 25

def draw_row(draw, y, col1, col2, font):
    # 좌우 정렬된 한 줄 그리기 (예: "합계금액          250,000")
    draw.text((30, y), col1, font=font, fill=(30, 30, 30))
    
    # col2는 오른쪽 정렬 (대략적 위치 계산)
    text_width = font.getlength(col2)
    draw.text((450 - text_width, y), col2, font=font, fill=(30, 30, 30))
    return y + 30

# ---------------------------------------------------------
# 2. 영수증 생성 로직
# ---------------------------------------------------------
# 👇 [수정] 학생 명단 불러오기 함수 추가
def load_student_db():
    try:
        with open("mock_data/student_db.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ 오류: 'mock_data/student_db.json' 파일이 없습니다. generate_student_db.py를 먼저 실행하세요.")
        exit()

# 전역 변수로 로드
STUDENT_DB = load_student_db()


def create_hardcore_receipt(index):
    # 1. 캔버스: 좁고 긴 감열지 스타일 (500 x 1000)
    width, height = 480, random.randint(800, 1100) # 길이는 랜덤
    # 배경색: 완전 흰색이 아니라 살짝 회색빛/누런빛 (현실감)
    bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 250))
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # 폰트 설정 (작은 글씨)
    font_s = ImageFont.truetype(FONT_PATH, 18) # 깨알 글씨
    font_m = ImageFont.truetype(FONT_PATH, 24) # 본문
    font_b = ImageFont.truetype(FONT_PATH, 30) # 강조
    font_h = ImageFont.truetype(FONT_PATH, 40) # 헤더

    # 👇 [수정] 랜덤 데이터 생성 -> DB에서 뽑아오기로 변경!
    target_student = random.choice(STUDENT_DB) # 명단에서 1명 랜덤 선택
    
    academy_name = random.choice(['빼어난수학', '서울아카데미', '연세입시학원']) + " 학원"
    owner_name = fake.name()
    biz_num = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10000,99999)}"
    date_time = fake.date_time_this_year().strftime("%y/%m/%d %H:%M:%S")
    card_num = f"{random.randint(4000,4999)}-****-****-{random.randint(1000,9999)}"
    approval_num = str(random.randint(10000000, 99999999))
    
    # 금액 생성 (수강료 + 교재비 등)
    items = []
    total_price = 0
    
    # 90% 확률로 수강료 청구
    if random.random() < 0.9:
        fee = target_student['base_fee'] # 🔥 DB 값 사용
        items.append((f"수강료({target_student['course_name']})", fee))
        total_price += fee
        
    # 50% 확률로 교재비 청구 (교재비가 있는 학생만)
    if target_student['book_fee'] > 0 and random.random() < 0.5:
        book = target_student['book_fee'] # 🔥 DB 값 사용
        items.append(("교재비", book))
        total_price += book

    if total_price == 0: # 둘 다 안 걸렸으면 수강료 강제 추가
        fee = target_student['base_fee']
        items.append((f"수강료({target_student['course_name']})", fee))
        total_price += fee

    vat = int(total_price * 0.1)
    supply_price = total_price - vat

    # --- 그리기 시작 ---
    y = 40
    
    # [헤더]
    draw.text((100, y), "[신용카드 매출전표]", font=font_m, fill=(0,0,0)); y += 40
    draw.text((30, y), "(고객용)", font=font_s, fill=(0,0,0))
    draw.text((350, y), "(주)스마트로", font=font_s, fill=(0,0,0)); y += 30
    
    y = draw_dashed_line(draw, y, width)
    
    # [가맹점 정보]
    draw.text((30, y), academy_name, font=font_b, fill=(0,0,0)); y += 35
    y = draw_row(draw, y, f"사업자: {biz_num}", f"대표: {owner_name}", font=font_s)
    y = draw_row(draw, y, "TEL: 031-123-4567", "", font=font_s)
    draw.text((30, y), f"주소: {fake.address()}", font=font_s, fill=(0,0,0)); y += 30
    
    y = draw_dashed_line(draw, y, width)
    
    # [결제 정보] (표 형식처럼 정렬)
    y = draw_row(draw, y, "거래일시:", date_time, font=font_m)
    y = draw_row(draw, y, "카드번호:", card_num, font=font_m)
    y = draw_row(draw, y, "승인번호:", approval_num, font=font_m)
    y = draw_row(draw, y, "할부개월:", "일시불", font=font_m)
    
    y = draw_dashed_line(draw, y, width)
    
    # [금액 상세] (여기가 중요: 표 형식)
    # 헤더
    draw.text((30, y), "품목      단가    수량    금액", font=font_s, fill=(0,0,0)); y += 25
    
    # 바디
    for name, price in items:
        line_text = f"{name[:4]}   {price}   1   {price}" # 간단하게 구현
        draw.text((30, y), line_text, font=font_m, fill=(0,0,0)); y += 30

    y = draw_dashed_line(draw, y, width)

    y = draw_row(draw, y, "공급가액:", f"{supply_price:,}", font=font_m)
    y = draw_row(draw, y, "부가세:", f"{vat:,}", font=font_m)
    y = draw_dashed_line(draw, y, width)
    
    # [최종 금액] (가장 크게)
    y = draw_row(draw, y, "합 계:", f"{total_price:,} 원", font=font_b)

    y += 40
    
    # [손글씨 영역] (이모님 스타일)
    draw.text((30, y), "* 50000원 이상 할부 거래", font=font_s, fill=(100,100,100)); y += 40
    
    # 손글씨 느낌 (폰트를 다르게 하거나 색상을 파란색/검은색 볼펜처럼)
    # (여기서는 같은 폰트 쓰지만, 실제론 손글씨 폰트(NanumPen) 등을 쓰면 더 좋음)
    student_name = fake.name()
    draw.text((100, y), f"{student_name} (중등반)", font=font_h, fill=(0, 0, 150)) # 파란 볼펜 느낌

    # --- 라벨 데이터 저장 (Training용) ---
    # (좌표 BBox는 생략했지만, 실제 학습 땐 필요)
    label_data = {
        "total_amount": total_price,
        "student_name": target_student['name'],
        "student_id": target_student['id'], # ID도 저장해두면 확실함
        "date": date_time.split()[0]
    }
    # ---------------------------------------------------------
    # 3. Hardcore Augmentation (여기가 핵심!)
    # ---------------------------------------------------------
    
    image_np = np.array(image)

    transform = A.Compose([
        # 1. 물리적 왜곡 (구겨짐 효과)
        #    ElasticTransform: 종이가 쭈글쭈글해지는 느낌
        #    GridDistortion: 종이가 울퉁불퉁한 느낌
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.8),

        # 2. 원근감 (Perspective) - 책상 위에 놓인 듯 비스듬하게
        A.Perspective(scale=(0.05, 0.1), p=0.7),

        # 3. 화질 저하 (카메라 초점 나감, 흔들림)
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.ImageCompression(quality_lower=30, quality_upper=70, p=0.5), # JPG 압축 노이즈
        ], p=0.6),

        # 4. 조명 및 그림자 (Shadow) - 가장 중요!
        #    종이 전체가 균일하게 밝지 않고, 그림자가 져야 리얼함
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1), 
            num_shadows_lower=1, 
            num_shadows_upper=3, 
            shadow_dimension=5, 
            p=0.7
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),

        # 5. 회전 (똑바로 찍지 않음)
        A.SafeRotate(limit=10, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=(200, 200, 200)), # 회전 후 빈공간 회색 처리
    ])

    augmented = transform(image=image_np)['image']
    final_image = Image.fromarray(augmented)

    # 5. 저장
    filename = f"hardcore_receipt_{index:05d}"
    final_image.save(f"{DATASET_DIR}/images/{filename}.jpg")
    
    with open(f"{DATASET_DIR}/labels/{filename}.json", 'w', encoding='utf-8') as f:
        json.dump(label_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("🔥 하드코어 영수증 생성 시작...")
    for i in range(20): # 테스트로 20장만
        create_hardcore_receipt(i)
    print("✅ 생성 완료! 'dataset/hardcore_train' 폴더를 확인하세요.")