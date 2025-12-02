import os
import random
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import Faker
import albumentations as A

# 설정
DATASET_DIR = "dataset/multi_receipt_train"
os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels", exist_ok=True)

fake = Faker('ko_KR')
FONT_PATH = "NanumGothic.ttf"

# 학생 DB 로드
try:
    with open("mock_data/student_db.json", "r", encoding="utf-8") as f:
        STUDENT_DB = json.load(f)
except FileNotFoundError:
    # DB가 없으면 임시 데이터 생성
    STUDENT_DB = [{"name": "홍길동", "course_name": "수학", "base_fee": 250000, "book_fee": 20000}]

# ---------------------------------------------------------
# 유틸리티 함수: 랜덤 날짜/금액 포맷
# ---------------------------------------------------------
def get_random_date_str():
    dt = fake.date_time_this_year()
    formats = [
        "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%Y.%m.%d %H:%M",
        "%y-%m-%d %H:%M", "%Y년 %m월 %d일", "%m/%d %H:%M"
    ]
    return dt.strftime(random.choice(formats)), dt.strftime("%Y-%m-%d")

def format_price(price):
    if random.random() < 0.2: return f"{price}" # 콤마 없음
    if random.random() < 0.5: return f"{price:,}" # 콤마만
    return f"{price:,}원" # 원 포함

# ---------------------------------------------------------
# [스타일 1] 표준 학원 영수증 (기존 스타일)
# ---------------------------------------------------------
def draw_receipt_type_standard(draw, width, items, total_price, student_name, academy_name, date_str, fonts):
    y = 30
    font_s, font_m, font_b = fonts

    # 헤더
    draw.text((width//2 - 80, y), "[영수증]", font=font_b, fill=(0,0,0))
    y += 50
    draw.text((20, y), f"가맹점: {academy_name}", font=font_m, fill=(0,0,0))
    y += 30
    draw.text((20, y), f"일시: {date_str}", font=font_s, fill=(50,50,50))
    y += 40
    
    draw.line((20, y, width-20, y), fill=(0,0,0), width=2)
    y += 20
    
    # 품목
    for name, price in items:
        draw.text((20, y), name, font=font_m, fill=(0,0,0))
        p_text = f"{price:,}"
        w = font_m.getlength(p_text)
        draw.text((width-20-w, y), p_text, font=font_m, fill=(0,0,0))
        y += 30
        
    y += 20
    draw.line((20, y, width-20, y), fill=(0,0,0), width=1)
    y += 20
    
    # 합계
    draw.text((20, y), "합  계", font=font_b, fill=(0,0,0))
    total_text = f"{total_price:,}"
    w = font_b.getlength(total_text)
    draw.text((width-20-w, y), total_text, font=font_b, fill=(0,0,0))
    y += 60
    
    # 학생 이름 (손글씨 느낌 또는 인쇄)
    draw.text((width//2 - 30, y), f"학생: {student_name}", font=font_m, fill=(0,0,0))
    
    return y + 50 # 최종 높이 반환

# ---------------------------------------------------------
# [스타일 2] 신용카드 매출전표 스타일 (점선, 카드번호 등)
# ---------------------------------------------------------
def draw_receipt_type_card(draw, width, items, total_price, student_name, academy_name, date_str, fonts):
    y = 20
    font_s, font_m, font_b = fonts
    
    # 헤더
    draw.text((width//2 - 100, y), "신용카드 매출전표", font=font_b, fill=(0,0,0))
    y += 40
    draw.text((width//2 - 60, y), "(회원용)", font=font_s, fill=(0,0,0))
    y += 40
    
    # 가맹점 정보
    draw.text((20, y), f"가맹점명 : {academy_name}", font=font_s, fill=(0,0,0))
    y += 25
    draw.text((20, y), f"사업자번호: {fake.business_number()}", font=font_s, fill=(0,0,0))
    y += 25
    draw.text((20, y), f"전화번호 : {fake.phone_number()}", font=font_s, fill=(0,0,0))
    y += 35
    
    # 점선 그리기 함수
    def draw_dashed_line(yy):
        for x in range(20, width-20, 10):
            draw.line((x, yy, x+5, yy), fill=(0,0,0), width=1)

    draw_dashed_line(y)
    y += 20
    
    # 품목 (간략하게)
    draw.text((20, y), "품명", font=font_s, fill=(0,0,0))
    draw.text((width-80, y), "금액", font=font_s, fill=(0,0,0))
    y += 25
    
    for name, price in items:
        # 긴 이름 자르기
        if len(name) > 10: name = name[:10] + "..."
        draw.text((20, y), name, font=font_m, fill=(0,0,0))
        p_text = f"{price:,}"
        w = font_m.getlength(p_text)
        draw.text((width-20-w, y), p_text, font=font_m, fill=(0,0,0))
        y += 30

    draw_dashed_line(y)
    y += 20
    
    # 합계 (강조)
    draw.text((20, y), "합 계 금 액", font=font_b, fill=(0,0,0))
    total_text = f"{total_price:,}원"
    w = font_b.getlength(total_text)
    draw.text((width-20-w, y), total_text, font=font_b, fill=(0,0,0))
    y += 50
    
    # 카드 정보 (가짜)
    card_num = f"{random.randint(1000,9999)}-****-****-{random.randint(1000,9999)}"
    draw.text((20, y), f"카드번호: {card_num}", font=font_s, fill=(0,0,0))
    y += 25
    draw.text((20, y), f"승인일시: {date_str}", font=font_s, fill=(0,0,0))
    y += 40
    
    # 학생 이름 (메모란)
    draw.rectangle((20, y, width-20, y+40), outline=(0,0,0), width=1)
    draw.text((30, y+10), f"원생: {student_name}", font=font_m, fill=(0,0,0))
    
    return y + 60

# ---------------------------------------------------------
# [스타일 3] 간이 영수증 (테두리 박스)
# ---------------------------------------------------------
def draw_receipt_type_simple(draw, width, items, total_price, student_name, academy_name, date_str, fonts):
    y = 40
    font_s, font_m, font_b = fonts
    
    # 외곽 테두리
    # 높이는 나중에 crop으로 해결되므로 일단 넉넉하게 그리지 않음 (함수 밖에서 처리 불가하므로 디자인 요소로만)
    
    # 타이틀
    draw.text((width//2 - 60, y), "간이영수증", font=font_b, fill=(0,0,0))
    y += 60
    
    # 공급자
    draw.text((30, y), f"공급자: {academy_name} (인)", font=font_m, fill=(0,0,0))
    y += 40
    
    # 날짜
    draw.text((30, y), f"작성일: {date_str}", font=font_s, fill=(0,0,0))
    y += 40
    
    # 테이블 헤더
    draw.rectangle((20, y, width-20, y+30), fill=(220,220,220))
    draw.text((30, y+5), "품목", font=font_s, fill=(0,0,0))
    draw.text((width-100, y+5), "금액", font=font_s, fill=(0,0,0))
    y += 40
    
    # 내용
    for name, price in items:
        draw.text((30, y), name, font=font_m, fill=(0,0,0))
        p_text = f"{price:,}"
        w = font_m.getlength(p_text)
        draw.text((width-30-w, y), p_text, font=font_m, fill=(0,0,0))
        draw.line((20, y+25, width-20, y+25), fill=(200,200,200), width=1)
        y += 35
        
    y += 20
    # 영수 금액
    draw.text((30, y), "위 금액을 영수함", font=font_m, fill=(0,0,0))
    y += 40
    
    # 총액 박스
    draw.rectangle((20, y, width-20, y+50), outline=(0,0,0), width=2)
    draw.text((40, y+15), f"Total: {total_price:,}", font=font_b, fill=(0,0,0))
    
    # 이름 (우측 하단)
    y += 70
    draw.text((width-150, y), f"성명: {student_name}", font=font_m, fill=(0,0,0))
    
    return y + 50

# ---------------------------------------------------------
# 통합 생성 함수
# ---------------------------------------------------------
def generate_single_receipt_content():
    width = 400
    # 높이는 넉넉하게 잡고 나중에 자름
    temp_height = 1000 
    
    # 종이 색상 랜덤 (흰색, 누런색, 푸른색, 회색)
    bg_choices = [
        (255, 255, 255), (250, 250, 240), (240, 248, 255), (245, 245, 245)
    ]
    bg_color = random.choice(bg_choices)
    
    image = Image.new('RGBA', (width, temp_height), color=bg_color + (255,))
    draw = ImageDraw.Draw(image)
    
    # 데이터 준비
    target_student = random.choice(STUDENT_DB)
    academy_name = random.choice(['수학의정석', '하이퍼매쓰', 'SKY입시', '청담어학원'])
    date_str, date_iso = get_random_date_str()
    
    items = []
    total_price = 0
    if random.random() < 0.9:
        items.append((f"수강료({target_student['course_name']})", target_student['base_fee']))
        total_price += target_student['base_fee']
    if target_student['book_fee'] > 0 and random.random() < 0.5:
        items.append(("교재비", target_student['book_fee']))
        total_price += target_student['book_fee']
    if total_price == 0:
        items.append(("교육비", target_student['base_fee']))
        total_price += target_student['base_fee']

    # 폰트 로드
    font_s = ImageFont.truetype(FONT_PATH, 18)
    font_m = ImageFont.truetype(FONT_PATH, 22)
    font_b = ImageFont.truetype(FONT_PATH, 28)
    fonts = (font_s, font_m, font_b)

    # 스타일 랜덤 선택
    style_choice = random.choice(['standard', 'card', 'simple'])
    
    final_y = 0
    if style_choice == 'standard':
        final_y = draw_receipt_type_standard(draw, width, items, total_price, target_student['name'], academy_name, date_str, fonts)
    elif style_choice == 'card':
        final_y = draw_receipt_type_card(draw, width, items, total_price, target_student['name'], academy_name, date_str, fonts)
    else:
        final_y = draw_receipt_type_simple(draw, width, items, total_price, target_student['name'], academy_name, date_str, fonts)

    # 이미지 자르기 (내용 있는 만큼만)
    image = image.crop((0, 0, width, final_y))
    
    # 라벨 정보 (좌표 제거됨, 학습용 텍스트만)
    label_info = {
        "student": target_student['name'],
        "amount": total_price,
        "date": date_iso,
        "items": items
    }
    
    return image, label_info

# ---------------------------------------------------------
# 증강 및 배치 (기존 로직 유지 + 강화)
# ---------------------------------------------------------
def apply_crumple_effect(pil_img):
    img_np = np.array(pil_img)
    transform = A.Compose([
        # 왜곡, 블러, 노이즈, 밝기 변화 등 다양화
        A.ElasticTransform(alpha=40, sigma=40*0.05, alpha_affine=40*0.03, p=0.8),
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        A.ISONoise(p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
    ])
    augmented = transform(image=img_np)['image']
    return Image.fromarray(augmented, 'RGBA')

def create_multi_receipt_scene(index):
    bg_width, bg_height = 1024, 1024
    desk_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    background = Image.new('RGBA', (bg_width, bg_height), color=desk_color + (255,))
    
    bg_np = np.array(background)
    noise = np.random.randint(-20, 20, bg_np.shape, dtype='int16')
    bg_np = np.clip(bg_np + noise, 0, 255).astype('uint8')
    background = Image.fromarray(bg_np, 'RGBA')

    receipts_metadata = []
    # 1장에 집중 학습하도록 1개만 배치 (복잡도 낮춤)
    num_receipts = 1 
    
    for _ in range(num_receipts):
        receipt_img, metadata = generate_single_receipt_content()
        receipt_img = apply_crumple_effect(receipt_img)
        
        # 회전 및 위치
        angle = random.uniform(-10, 10)
        receipt_img = receipt_img.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        w, h = receipt_img.size
        x = (bg_width - w) // 2 + random.randint(-50, 50)
        y = (bg_height - h) // 2 + random.randint(-50, 50)
        
        # 그림자
        shadow = Image.new('RGBA', receipt_img.size, (0, 0, 0, 0))
        shadow_np = np.array(shadow)
        mask = np.array(receipt_img)[:, :, 3] > 0
        shadow_np[mask] = [0, 0, 0, 100]
        shadow = Image.fromarray(shadow_np).filter(ImageFilter.GaussianBlur(10))
        
        background.paste(shadow, (x+10, y+10), shadow)
        background.paste(receipt_img, (x, y), receipt_img)
        
        receipts_metadata.append(metadata)
    
    final_image = background.convert('RGB')
    filename = f"multi_receipt_{index:05d}"
    
    # 학습/검증용 폴더 분리 없이 다 때려넣음 (Kaggle에서 split 함)
    final_image.save(f"{DATASET_DIR}/images/{filename}.jpg")
    
    with open(f"{DATASET_DIR}/labels/{filename}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "file": f"{filename}.jpg",
            "receipts": receipts_metadata
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("🔥 업그레이드된 영수증 데이터 생성 시작 (다양한 레이아웃)...")
    for i in range(1000): # 1000장 생성
        create_multi_receipt_scene(i)
        if (i+1) % 100 == 0: print(f"{i+1}장 완료...")
    print("✅ 생성 완료!")