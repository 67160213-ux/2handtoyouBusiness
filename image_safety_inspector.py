# -*- coding: utf-8 -*-
"""
image_safety_inspector.py
-------------------------------------------------------------------
Local Image Safety Inspector Module (Pillow / PIL Based)
สำหรับประมวลผลและตรวจสอบความปลอดภัยของไฟล์รูปภาพสินค้ามือสอง
- ตรวจสอบความถูกต้องของไฟล์รูปภาพ (File Header, Resolution, Corruption)
- ตรวจสอบรูปภาพสแปม/ภาพเปล่า/ภาพสีเดียว (Blank / Solid Color Placeholder)
- ตรวจสอบรูปภาพต้องสงสัย (Skin Tone / Explicit Content Ratio Heuristics)
- ตรวจสอบรูปภาพซ้ำซ้อน (Duplicate Image Hashing)
-------------------------------------------------------------------
"""

import io
import math
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageStat

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def calculate_image_hash(image: Image.Image, hash_size: int = 8) -> str:
    """
    คำนวณ perceptual difference hash (dhash) สำหรับตรวจหาภาพซ้ำซ้อน
    """
    try:
        resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
        pixels = list(resized.getdata())
        difference = []
        for row in range(hash_size):
            for col in range(hash_size):
                pixel_left = pixels[row * (hash_size + 1) + col]
                pixel_right = pixels[row * (hash_size + 1) + col + 1]
                difference.append(pixel_left > pixel_right)
        decimal_value = 0
        hex_string = []
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2 ** (index % 4)
            if (index % 4) == 3:
                hex_string.append(hex(decimal_value)[2:])
                decimal_value = 0
        return "".join(hex_string)
    except Exception:
        return ""

def analyze_skin_ratio(image: Image.Image) -> float:
    """
    คำนวณสัดส่วนสีผิว (Skin-Tone Ratio) เบื้องต้นในระบบ HSV เพื่อตรวจหาภาพสุ่มเสี่ยง NSFW
    """
    try:
        rgb_img = image.convert("RGB").resize((100, 100))
        pixels = list(rgb_img.getdata())
        skin_count = 0
        total_pixels = len(pixels)

        for r, g, b in pixels:
            # Rule for skin tone in RGB color space
            if (r > 95) and (g > 40) and (b > 20) and ((max(r, g, b) - min(r, g, b)) > 15) and (abs(r - g) > 15) and (r > g) and (r > b):
                skin_count += 1

        return skin_count / total_pixels
    except Exception:
        return 0.0

def inspect_single_image(file_bytes: bytes, filename: str = "image.jpg") -> Dict[str, Any]:
    """
    วิเคราะห์ไฟล์รูปภาพ 1 ไฟล์ และส่งคืนผลการตรวจสอบความปลอดภัย
    """
    reasons = []
    is_safe = True
    risk_level = "LOW"
    status = "APPROVED"

    if not file_bytes or len(file_bytes) == 0:
        return {
            "is_safe": False,
            "risk_level": "HIGH",
            "status": "BLOCKED",
            "reasons": ["ไฟล์รูปภาพมีความยาว 0 bytes (ไฟล์เสีย)"],
            "img_hash": ""
        }

    if len(file_bytes) > MAX_FILE_SIZE:
        return {
            "is_safe": False,
            "risk_level": "MEDIUM",
            "status": "PENDING_REVIEW",
            "reasons": [f"ขนาดไฟล์รูปภาพเกินขีดจำกัด ({len(file_bytes) / (1024*1024):.1f} MB > 10 MB)"],
            "img_hash": ""
        }

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        
        # Re-open after verify (PIL requirement)
        img = Image.open(io.BytesIO(file_bytes))
        fmt = img.format.upper() if img.format else "UNKNOWN"
        width, height = img.size

        # 1. ตรวจประเภทไฟล์
        if fmt not in ALLOWED_FORMATS:
            return {
                "is_safe": False,
                "risk_level": "MEDIUM",
                "status": "PENDING_REVIEW",
                "reasons": [f"รูปแบบไฟล์รูปภาพไม่ได้รับอนุญาต ({fmt})"],
                "img_hash": ""
            }

        # 2. ตรวจขนาดความละเอียด (Resolution)
        if width < 50 or height < 50:
            reasons.append(f"ความละเอียดรูปภาพต่ำเกินไป ({width}x{height} px)")
            risk_level = "MEDIUM"
            status = "PENDING_REVIEW"
            is_safe = False

        # 3. ตรวจสอบ Aspect Ratio ผิดปกติเกิน 1:10
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > 10.0:
            reasons.append(f"สัดส่วนรูปภาพผิดปกติเกินไป (Aspect Ratio {aspect_ratio:.1f}:1)")
            risk_level = "MEDIUM"
            status = "PENDING_REVIEW"
            is_safe = False

        # 4. ตรวจภาพเปล่า/ภาพสีเดียว (Blank / Solid Color Image)
        stat = ImageStat.Stat(img.convert("L"))
        stddev = stat.stddev[0] if stat.stddev else 0
        if stddev < 3.0:
            reasons.append("รูปภาพเป็นภาพสีว่างเปล่าหรือไม่มีรายละเอียดสินค้า (Blank/Solid Color Image)")
            risk_level = "MEDIUM"
            status = "PENDING_REVIEW"
            is_safe = False

        # 5. ตรวจสัดส่วน Skin Tone / Content Safety
        skin_ratio = analyze_skin_ratio(img)
        if skin_ratio > 0.50:
            reasons.append(f"AI ตรวจพบสัดส่วนภาพสุ่มเสี่ยงโป๊เปลือย/NSFW (Skin Ratio: {skin_ratio*100:.1f}%)")
            risk_level = "HIGH"
            status = "BLOCKED"
            is_safe = False
        elif skin_ratio > 0.35:
            reasons.append(f"AI ตรวจพบภาพต้องสงสัยสุ่มเสี่ยง (Skin Ratio: {skin_ratio*100:.1f}%)")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
                status = "PENDING_REVIEW"
            is_safe = False

        # คำนวณ Hash สำหรับหาภาพซ้ำ
        img_hash = calculate_image_hash(img)

        return {
            "is_safe": is_safe,
            "risk_level": risk_level,
            "status": status,
            "reasons": reasons,
            "img_hash": img_hash,
            "metadata": {
                "format": fmt,
                "width": width,
                "height": height,
                "skin_ratio": round(skin_ratio, 4)
            }
        }

    except Exception as e:
        return {
            "is_safe": False,
            "risk_level": "MEDIUM",
            "status": "PENDING_REVIEW",
            "reasons": [f"ไม่สามารถอ่านไฟล์รูปภาพได้ ({str(e)})"],
            "img_hash": ""
        }

def inspect_multiple_images(images_list: List[Tuple[bytes, str]]) -> Dict[str, Any]:
    """
    วิเคราะห์รูปภาพหลายรูป พร้อมตรวจจับภาพซ้ำซ้อน (Duplicate Images)
    """
    if not images_list or len(images_list) == 0:
        return {
            "overall_safe": True,
            "highest_risk_level": "LOW",
            "final_status": "APPROVED",
            "all_reasons": [],
            "processed_count": 0
        }

    highest_risk = "LOW"
    final_status = "APPROVED"
    all_reasons = []
    hashes = []
    risk_weights = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

    for index, (file_bytes, filename) in enumerate(images_list, 1):
        res = inspect_single_image(file_bytes, filename)
        
        # ตรวจภาพซ้ำ
        if res["img_hash"]:
            if res["img_hash"] in hashes:
                all_reasons.append(f"รูปภาพลำดับที่ {index} ({filename}) เป็นรูปซ้ำซ้อนกับรูปก่อนหน้า")
                if risk_weights["MEDIUM"] > risk_weights[highest_risk]:
                    highest_risk = "MEDIUM"
                    final_status = "PENDING_REVIEW"
            else:
                hashes.append(res["img_hash"])

        if res["reasons"]:
            for r in res["reasons"]:
                all_reasons.append(f"รูปภาพ {filename}: {r}")

        if risk_weights.get(res["risk_level"], 1) > risk_weights.get(highest_risk, 1):
            highest_risk = res["risk_level"]
            final_status = res["status"]

    return {
        "overall_safe": highest_risk == "LOW",
        "highest_risk_level": highest_risk,
        "final_status": final_status,
        "all_reasons": all_reasons,
        "processed_count": len(images_list)
    }
