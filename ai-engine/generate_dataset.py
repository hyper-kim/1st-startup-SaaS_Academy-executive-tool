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
    print("❌ 학생 DB가 없습니다. generate_student_db.py를 먼저 실행하세요.")
    exit()

# ---------------------------------------------------------
# 1. 단일 영수증 생성 (이미지 자체는 깔끔하게 생성 후 나중에 변형)
# ---------------------------------------------------------
def generate_single_receipt_content():
    # 너비는 고정, 높이는 내용에 따라 가변적
    width = 400
    padding = 40
    line_spacing = 30
    
    # 데이터 준비
    target_student = random.choice(STUDENT_DB)
    academy_name = random.choice(['수학의정석', '하이퍼매쓰', '서울아카데미', 'SKY입시']) + " 학원"
    date_time = fake.date_time_this_year().strftime("%Y-%m-%d %H:%M")
    
    items = []
    total_price = 0
    
    # 수강료/교재비 로직
    if random.random() < 0.9:
        items.append((f"수강료({target_student['course_name']})", target_student['base_fee']))
        total_price += target_student['base_fee']
    if target_student['book_fee'] > 0 and random.random() < 0.5:
        items.append(("교재비", target_student['book_fee']))
        total_price += target_student['book_fee']
    if total_price == 0:
        items.append(("수강료", target_student['base_fee']))
        total_price += target_student['base_fee']

    # 높이 계산 (항목 수에 따라)
    height = 350 + (len(items) * line_spacing) + 150
    
    # 약간 누런 종이 배경
    bg_color = (random.randint(245, 255), random.randint(245, 255), random.randint(240, 250))
    image = Image.new('RGBA', (width, height), color=bg_color + (255,))
    draw = ImageDraw.Draw(image)
    
    font_s = ImageFont.truetype(FONT_PATH, 18)
    font_m = ImageFont.truetype(FONT_PATH, 22)
    font_b = ImageFont.truetype(FONT_PATH, 28)

    y = 30
    # [헤더]
    draw.text((width//2 - 80, y), "[영수증]", font=font_b, fill=(0,0,0))
    y += 50
    draw.text((padding, y), f"가맹점: {academy_name}", font=font_m, fill=(0,0,0))
    y += 30
    draw.text((padding, y), f"일시: {date_time}", font=font_s, fill=(50,50,50))
    y += 40
    
    # 구분선
    draw.line((padding, y, width-padding, y), fill=(0,0,0), width=2)
    y += 20
    
    # [품목]
    for name, price in items:
        draw.text((padding, y), name, font=font_m, fill=(0,0,0))
        p_text = f"{price:,}"
        w = font_m.getlength(p_text)
        draw.text((width-padding-w, y), p_text, font=font_m, fill=(0,0,0))
        y += line_spacing
        
    y += 20
    draw.line((padding, y, width-padding, y), fill=(0,0,0), width=1)
    y += 20
    
    # [합계]
    draw.text((padding, y), "합  계", font=font_b, fill=(0,0,0))
    total_text = f"{total_price:,} 원"
    w = font_b.getlength(total_text)
    draw.text((width-padding-w, y), total_text, font=font_b, fill=(0,0,0))
    y += 60
    
    # [손글씨 이름] (파란 볼펜 느낌)
    pen_color = (0, 0, random.randint(100, 200))
    draw.text((width//2 - 30, y), target_student['name'], font=font_b, fill=pen_color)
    
    # JSON 라벨 정보 (상대 좌표는 나중에 절대 좌표로 변환)
    label_info = {
        "student": target_student['name'],
        "amount": total_price,
        "date": date_time.split()[0],
        "items": items
    }
    
    return image, label_info

# ---------------------------------------------------------
# 2. 종이 질감 및 구겨짐 효과 (Elastic Transform)
# ---------------------------------------------------------
def apply_crumple_effect(pil_img):
    # 1. PIL(RGBA) -> Numpy 변환
    img_np = np.array(pil_img) # (Height, Width, 4)

    # 2. 변형 정의
    # Albumentations는 기본적으로 다채널 이미지를 지원합니다.
    # RGBA 4채널을 통째로 넣어서, 형태(Alpha)와 색상(RGB)이 같이 구겨지게 합니다.
    transform = A.Compose([
        # 물리적 왜곡 (구겨짐) - 빈 공간은 투명하게(0) 채움
        A.ElasticTransform(
            alpha=60, 
            sigma=60 * 0.05, 
            alpha_affine=60 * 0.03, 
            p=1.0, 
            border_mode=cv2.BORDER_CONSTANT, 
            value=(0,0,0,0) # 투명 배경
        ),
        # 노이즈 추가 (RGB, Alpha 모두 약간씩 들어가도 무방함)
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
    ])
    
    # 3. 적용
    augmented = transform(image=img_np)['image']
    
    # 4. 결과 반환 (RGBA 모드 명시)
    return Image.fromarray(augmented, 'RGBA')

# ---------------------------------------------------------
# 3. 메인 생성기: 책상 위에 여러 장 배치 + 그림자
# ---------------------------------------------------------
def create_multi_receipt_scene(index):
    # 1. 배경 (책상) 생성 - 1024x1024
    bg_width, bg_height = 1024, 1024
    # 책상 색상 (나무색 or 회색 톤)
    desk_color = (random.randint(100, 150), random.randint(80, 130), random.randint(60, 100))
    background = Image.new('RGBA', (bg_width, bg_height), color=desk_color + (255,))
    
    # 배경에 노이즈 추가 (질감)
    bg_np = np.array(background)
    noise = np.random.randint(-20, 20, bg_np.shape, dtype='int16')
    bg_np = np.clip(bg_np + noise, 0, 255).astype('uint8')
    background = Image.fromarray(bg_np, 'RGBA')

    receipts_metadata = []
    
    # 영수증 개수 (1 ~ 3개 랜덤)
    num_receipts = random.randint(1, 3)
    
    # 겹치지 않게 배치하기 위한 영역 리스트
    occupied_boxes = []

    for _ in range(num_receipts):
        # 1. 영수증 생성
        receipt_img, metadata = generate_single_receipt_content()
        
        # 2. 구겨짐 효과 적용
        receipt_img = apply_crumple_effect(receipt_img)
        
        # 3. 회전 (아주 약간만, +/- 5도)
        angle = random.uniform(-5, 5)
        receipt_img = receipt_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(0,0,0,0)) # 투명 배경 확장
        
        # 4. 그림자 생성 (Drop Shadow)
        # 영수증 모양의 검은색 마스크 생성
        shadow = Image.new('RGBA', receipt_img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        # 영수증이 있는 부분(알파>0)만 검게 칠함
        r_np = np.array(receipt_img)
        mask = r_np[:, :, 3] > 0
        shadow_np = np.array(shadow)
        shadow_np[mask] = [0, 0, 0, 100] # 검은색, 투명도 100
        shadow = Image.fromarray(shadow_np)
        
        # 블러 처리로 그림자 부드럽게
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
        
        # 5. 위치 선정 (겹치지 않게 시도)
        w, h = receipt_img.size
        placed = False
        
        for _ in range(10): # 10번 시도
            x = random.randint(50, bg_width - w - 50)
            y = random.randint(50, bg_height - h - 50)
            
            # 겹침 확인 (간단한 박스 충돌)
            collision = False
            new_box = [x, y, x+w, y+h]
            for box in occupied_boxes:
                # 겹치는지 확인 (A.left < B.right and A.right > B.left ...)
                if (new_box[0] < box[2] and new_box[2] > box[0] and
                    new_box[1] < box[3] and new_box[3] > box[1]):
                    collision = True
                    break
            
            if not collision:
                occupied_boxes.append(new_box)
                
                # 6. 붙이기 (그림자 먼저, 그 위에 영수증)
                # 그림자는 약간 아래 오른쪽으로 치우치게 (+10, +10)
                background.paste(shadow, (x + 10, y + 10), shadow) 
                background.paste(receipt_img, (x, y), receipt_img)
                
                # 7. 메타데이터 좌표 업데이트 (절대 좌표)
                # (실제로는 회전된 텍스트 좌표 계산이 복잡하지만, 
                # 여기서는 OCR 학습용으로 '영수증의 내용'만 저장합니다.
                # LayoutLM 학습을 위해서는 BBox 계산이 더 정교해야 함)
                metadata['position'] = {"x": x, "y": y, "w": w, "h": h}
                receipts_metadata.append(metadata)
                placed = True
                break
    
    # 8. 최종 저장 (JPG로 변환하여 배경과 합침)
    final_image = background.convert('RGB')
    
    filename = f"multi_receipt_{index:05d}"
    final_image.save(f"{DATASET_DIR}/images/{filename}.jpg")
    
    with open(f"{DATASET_DIR}/labels/{filename}.json", 'w', encoding='utf-8') as f:
        json.dump({
            "file": f"{filename}.jpg",
            "receipts": receipts_metadata
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    print("🔥 멀티 영수증 데이터셋 생성 시작...")
    # 100장 생성
    for i in range(100):
        create_multi_receipt_scene(i)
        if (i+1) % 10 == 0: print(f"{i+1}장 생성 완료...")
    print("✅ 생성 완료! 'dataset/multi_receipt_train' 폴더 확인.")