# -*- coding: utf-8 -*-
"""
test_api.py - สคริปต์ทดสอบยิง API Local AI Moderation & Suggestion Server
"""

import sys
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://127.0.0.1:8000"

# 1. Test Endpoint 1: Inspect Listing
inspect_url = f"{BASE_URL}/api/v1/ai/inspect-listing"
test_cases_inspect = [
    {
        "name": "กรณีปืนและกระสุน (Weapon)",
        "data": {
            "title": "ขายปืนสั้น Unregistered Handgun มือสอง",
            "description": "สภาพ 90% แถมกระสุนซ้อม 50 นัด"
        }
    },
    {
        "name": "กรณีเอกสารปลอม (Illegal Docs)",
        "data": {
            "title": "รับทำ Fake Id Card และวุฒิการศึกษา",
            "description": "ส่งด่วนได้ของแน่นอน แบงค์ปลอมก็มี"
        }
    },
    {
        "name": "กรณีสินค้าละเมิดลิขสิทธิ์ (Counterfeit)",
        "data": {
            "title": "ขายกระเป๋าแบรนด์ปลอม เกรดมิลเลอร์ งานก๊อป",
            "description": "เหมือนแท้ที่สุดในตลาด หนังอย่างดี"
        }
    },
    {
        "name": "กรณีสินค้าปกติ (Normal Product)",
        "data": {
            "title": "ขายมือถือ Xiaomi Redmi Note 13 มือสอง",
            "description": "อุปกรณ์ครบกล่อง สภาพดีมาก ไม่มีรอยถลอก"
        }
    }
]

print("=" * 60)
print("🧪 [1/2] เริ่มทดสอบ Endpoint 1: /api/v1/ai/inspect-listing")
print("=" * 60)

import io
from PIL import Image

def generate_sample_image(color=(200, 150, 100), size=(300, 300)):
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()

for test in test_cases_inspect:
    print(f"\n🔹 {test['name']}")
    try:
        response = requests.post(inspect_url, data=test["data"])
        if response.status_code == 200:
            res = response.json()
            print("  Result 3 ค่าที่ส่งให้ SE:")
            print(f"   1. หมวดหมู่ (category)  : {res['result']['category']}")
            print(f"   2. ความเสี่ยง (risk_level): {res['result']['risk_level']}")
            print(f"   3. สถานะ (status)      : {res['result']['status']}")
            print(f"   Confidence Score       : {res['metadata']['confidence_score']}")
            print(f"   Reasons                : {res['metadata']['detected_reasons']}")
        else:
            print("  ⚠️ HTTP Error:", response.status_code)
    except Exception as e:
        print("  ⚠️ เกิดข้อผิดพลาด:", e)

# 1.1 Test Endpoint 1 with Image Upload
print("\n🔹 กรณีทดสอบอัปโหลดรูปภาพสินค้า (Image Inspection Test)")
try:
    img_bytes = generate_sample_image(color=(255, 100, 100))
    files = [("images", ("test_product.jpg", img_bytes, "image/jpeg"))]
    data = {
        "title": "ขายกล้องมือสอง Canon EOS 80D",
        "description": "สภาพสวย ใช้งานได้ปกติ แถมกระเป๋ากล้อง"
    }
    response = requests.post(inspect_url, data=data, files=files)
    if response.status_code == 200:
        res = response.json()
        print("  Result 3 ค่าที่ส่งให้ SE (มีรูปภาพ):")
        print(f"   1. หมวดหมู่ (category)  : {res['result']['category']}")
        print(f"   2. ความเสี่ยง (risk_level): {res['result']['risk_level']}")
        print(f"   3. สถานะ (status)      : {res['result']['status']}")
        print(f"   จำนวนรูปภาพที่ส่ง        : {res['metadata']['images_received_count']}")
        print(f"   Reasons                : {res['metadata']['detected_reasons']}")
    else:
        print("  ⚠️ HTTP Error:", response.status_code)
except Exception as e:
    print("  ⚠️ เกิดข้อผิดพลาดในการทดสอบอัปโหลดรูปภาพ:", e)

# 2. Test Endpoint 2: Suggest Description
suggest_url = f"{BASE_URL}/api/v1/ai/suggest-description"
test_cases_suggest = [
    {
        "name": "แนะนำคำอธิบายรองเท้า",
        "json": {
            "title": "ขายรองเท้า Nike Air Jordan 1",
            "condition": "USED_LIKE_NEW",
            "user_notes": "ใส่น้อยครั้ง มีกล่องครบ Size 42"
        }
    }
]

print("\n" + "=" * 60)
print("🧪 [2/2] เริ่มทดสอบ Endpoint 2: /api/v1/ai/suggest-description")
print("=" * 60)

for test in test_cases_suggest:
    print(f"\n🔹 {test['name']}")
    try:
        response = requests.post(suggest_url, json=test["json"])
        if response.status_code == 200:
            res = response.json()
            print(f"   Suggested Title : {res['suggested_title']}")
            print(f"   Suggested Description :\n{res['suggested_description']}")
            print(f"   Is Safe : {res['safety_check']['is_safe']}")
        else:
            print("  ⚠️ HTTP Error:", response.status_code)
    except Exception as e:
        print("  ⚠️ เกิดข้อผิดพลาด:", e)

print("=" * 60)

