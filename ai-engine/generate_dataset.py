import os
import random
import json
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import albumentations as A
from faker import Faker

# 설정
DATASET_DIR = "dataset/multi_receipt_train"
os.makedirs(f"{DATASET_DIR}/images", exist_ok=True)
os.makedirs(f"{DATASET_DIR}/labels", exist_ok=True)

fake = Faker('ko_KR')

# 1. 폰트 로드 (폰트가 없으면 에러나니 꼭 fonts 폴더 확인)
FONT_GOTHIC = "fonts/NanumGothic.ttf"
HAND_FONTS = glob.glob("fonts/*.ttf")
if not HAND_FONTS: HAND_FONTS = [FONT_GOTHIC]

def get_random_font(size):
    # 손글씨 중에서도 좀 두껍거나 휘갈기는 폰트를 선호하도록 로직 구성 가능
    font_path = random.choice(HAND_FONTS)
    try:
        return ImageFont.truetype(font_path, size)
    except:
        return ImageFont.truetype(FONT_GOTHIC, size)

def get_gothic_font(size):
    return ImageFont.truetype(FONT_GOTHIC, size)

# 금액 표기법 (만원, 콤마 등)
def format_money(amount):
    if random.random() < 0.4: # 40% 확률로 '만원' 표기
        if amount >= 10000 and amount % 10000 == 0:
            return f"{amount // 10000}만원"
    return f"{amount:,}"

# ==============================================================================
# [스타일 A] 신용카드 매출전표 (정형)
# ==============================================================================
def draw_card_receipt(draw, width):
    # ... (기존과 동일, 생략 없이 사용하려면 이전 코드 참고하거나 아래처럼 간단히 구현)
    y = 40
    font_b = get_gothic_font(26)
    font_m = get_gothic_font(20)
    
    academy = random.choice(['SKY입시', '하이퍼매쓰', '청담어학원'])
    price = random.choice([250000, 300000, 180000])
    student = fake.name()
    date_str = fake.date_this_year().strftime("%Y-%m-%d")

    draw.text((width//2 - 60, y), "신용카드전표", font=font_b, fill=0)
    y += 50
    draw.text((20, y), f"가맹점: {academy}", font=font_m, fill=0)
    y += 30
    draw.text((20, y), f"금액: {price:,}원", font=font_b, fill=0)
    y += 40
    draw.text((20, y), f"일시: {date_str}", font=font_m, fill=0)
    y += 40
    draw.text((20, y), f"학생: {student}", font=font_m, fill=0)
    
    return y + 50, {"student": student, "amount": price, "date": date_str, "type": "card"}

# ==============================================================================
# [스타일 B] 간이 영수증 (표 형태)
# ==============================================================================
def draw_gani_receipt(draw, width):
    y = 40
    font_hand = get_random_font(28)
    font_frame = get_gothic_font(20)
    
    price = random.choice([140000, 200000, 80000])
    price_str = format_money(price)
    student = fake.name()
    date_str = f"2025. {random.randint(1,12)}. {random.randint(1,28)}"

    draw.text((width//2 - 50, y), "영 수 증", font=get_gothic_font(26), fill=0)
    y += 50
    
    # 박스 그리기
    draw.rectangle((20, y, width-20, y+250), outline=0, width=2)
    draw.line((100, y, 100, y+250), fill=0, width=1) # 세로선
    draw.line((20, y+80, width-20, y+80), fill=0, width=1) # 가로선
    draw.line((20, y+160, width-20, y+160), fill=0, width=1) # 가로선

    # 라벨
    draw.text((35, y+30), "성 명", font=font_frame, fill=0)
    draw.text((35, y+110), "금 액", font=font_frame, fill=0)
    draw.text((35, y+190), "날 짜", font=font_frame, fill=0)

    # 손글씨 내용 (위치 약간씩 랜덤하게 비틀기)
    draw.text((120 + random.randint(-5,5), y+25 + random.randint(-5,5)), student, font=font_hand, fill=(0,0,50))
    draw.text((120 + random.randint(-5,5), y+105 + random.randint(-5,5)), price_str, font=font_hand, fill=(0,0,50))
    draw.text((120 + random.randint(-5,5), y+185 + random.randint(-5,5)), date_str, font=font_hand, fill=(0,0,50))

    return y + 300, {"student": student, "amount": price, "date": date_str, "type": "gani"}

# ==============================================================================
# [스타일 C] ★ 메모장/포스트잇 스타일 (양식 파괴)
# ==============================================================================
def draw_memo_receipt(draw, width):
    y = 40
    # 아주 큰 손글씨 (휘갈겨 쓴 느낌)
    font_big = get_random_font(40) 
    font_small = get_random_font(24)

    price = random.choice([250000, 300000, 450000, 50000])
    price_str = format_money(price)
    student = fake.name()
    
    # 날짜 포맷도 대충 (11/5, 11월 5일 등)
    m = random.randint(1, 12)
    d = random.randint(1, 30)
    if random.random() < 0.5:
        date_str = f"{m}/{d}"
    else:
        date_str = f"{m}월 {d}일"

    # 배치도 내맘대로 (줄바꿈 없이 쓰거나, 대각선으로 쓰거나)
    # 여기서는 간단히 줄바꿈만 랜덤으로
    
    # 1. 금액 (가장 크게)
    draw.text((40 + random.randint(-10,10), y), price_str, font=font_big, fill=(0,0,0))
    y += 60
    
    # 2. 학생 이름 (툭 던져놓기)
    draw.text((width - 150 + random.randint(-20,20), y), student, font=font_small, fill=(50,50,50))
    y += 40
    
    # 3. 날짜 (구석에)
    draw.text((40, y + 20), date_str, font=font_small, fill=(100,100,100))
    
    # 4. 기타 낙서 (완료, 입금됨 등)
    if random.random() < 0.5:
        draw.text((width - 100, y + 30), random.choice(["완료", "입금", "O"]), font=font_small, fill=(200,0,0))

    return y + 100, {"student": student, "amount": price, "date": f"2025-{m:02d}-{d:02d}", "type": "memo"}


# ==============================================================================
# 통합 생성기 (노이즈 강화)
# ==============================================================================
def create_receipt_image(index):
    width = 450
    height = 600 # 넉넉하게
    
    # 배경색: 흰색, 노란색(포스트잇), 갱지색 랜덤
    bg_choices = [
        (255, 255, 255), (255, 255, 240), (240, 240, 230), (255, 250, 205)
    ]
    bg_color = random.choice(bg_choices)
    
    image = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # 스타일 랜덤 선택 (카드 30%, 간이 40%, 메모 30%)
    rand_style = random.random()
    if rand_style < 0.3:
        final_y, metadata = draw_card_receipt(draw, width)
    elif rand_style < 0.7:
        final_y, metadata = draw_gani_receipt(draw, width)
    else:
        final_y, metadata = draw_memo_receipt(draw, width)
        
    image = image.crop((0, 0, width, final_y))
    
    # --------------------------------------------------------------------------
    # ★ 핵심 기술: 글자 뭉개기 (Realism Augmentation)
    # --------------------------------------------------------------------------
    img_np = np.array(image)
    
    transform = A.Compose([
        # 1. 글자 비틀기 (악필 효과)
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.7),
        
        # 2. 잉크 번짐/흐림 효과 (Erosion/Dilation/Blur)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        
        # 3. 조명/노이즈
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        
        # 4. 회전 (사진 찍을 때 삐뚤어짐)
        A.Rotate(limit=10, p=1.0, border_mode=cv2.BORDER_REPLICATE)
    ])
    
    augmented = transform(image=img_np)['image']
    final_image = Image.fromarray(augmented)
    
    # 저장
    filename = f"receipt_{index:05d}"
    final_image.save(f"{DATASET_DIR}/images/{filename}.jpg")
    
    with open(f"{DATASET_DIR}/labels/{filename}.json", "w", encoding="utf-8") as f:
        json.dump({"file": f"{filename}.jpg", "receipts": [metadata]}, f, ensure_ascii=False)

if __name__ == "__main__":
    print("🔥 리얼리티 강화 데이터셋 생성 시작 (카드/간이/메모 + 악필효과)...")
    for i in range(5000): 
        create_receipt_image(i)
        if (i+1) % 500 == 0: print(f"{i+1}장 완료...")