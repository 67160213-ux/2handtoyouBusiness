# -*- coding: utf-8 -*-
"""
prepare_safety_dataset.py
-------------------------------------------------------------------
Script สำหรับเตรียมและฉลากข้อมูล (Data Labeling & Preparation Pipeline)
สำหรับระบบ AI ตรวจจับประกาศผิดปกติ (Prohibited & Illegal Product Moderation)

ผลลัพธ์:
สร้างคอลัมน์เป้าหมาย 3 คอลัมน์หลักตามสเปกธุรกิจ:
  1. ai_category   : WEAPON, DRUG, COUNTERFEIT, CONTROLLED_GOODS, WILDLIFE_ANIMAL,
                     ILLEGAL_DOCS, GAMBLING, HAZARDOUS, NORMAL
  2. ai_risk_level : HIGH, MEDIUM, LOW
  3. ai_status     : BLOCKED, PENDING_REVIEW, APPROVED
-------------------------------------------------------------------
"""

import os
import re
import sys
import pandas as pd

# ป้องกันปัญหา console encoding บน Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# กำหนดเส้นทางไฟล์
OUTPUT_DIR = "output"
SPLITS = ["train", "val", "test", "uat"]

# Rule Dictionaries สำหรับสแกนข้อความภาษาไทยและอังกฤษ
PROHIBITED_RULES = {
    "WEAPON": [
        r"ปืน", r"กระสุน", r"มีดผิดกฎหมาย", r"มีดพก", r"สนับมือ", r"ดัดแปลงอาวุธ",
        r"bb gun ดัดแปลง", r"handgun", r"firearm", r"weapon", r"rifle", r"pistol", r"shotgun", r"ammo"
    ],
    "DRUG": [
        r"ยาเสพติด", r"ยาบ้า", r"กัญชา", r"ไอซ์", r"บ้อง", r"เข็มฉีดยาเสพติด", r"กระท่อม",
        r"meth", r"cannabis", r"heroin", r"cocaine", r"narcotic", r"ecstasy"
    ],
    "ILLEGAL_DOCS": [
        r"ธนบัตรปลอม", r"แบงค์ปลอม", r"บัตรประชาชน", r" fake id", r"counterfeit currency",
        r"วุฒิการศึกษา", r"ใบขับขี่ปลอม", r"passport", r"เอกสารราชการ"
    ],
    "WILDLIFE_ANIMAL": [
        r"สัตว์ป่า", r"คุ้มครอง", r"ซากสัตว์", r"wildlife", r"protected species",
        r"งูพิษ", r"งาช้าง", r"นกขุนทอง", r"สัตว์ผิดกฎหมาย"
    ],
    "HAZARDOUS": [
        r"สารเคมี", r"วัตถุระเบิด", r"พลุ", r"ประทัดรุนแรง", r" explosive",
        r"chemical", r"acid", r"สารพิษ"
    ],
    "GAMBLING": [
        r"ตู้สล็อต", r"โพยหวย", r"อุปกรณ์พนัน", r" gambling", r"slot machine",
        r"casino chip", r"ไฮโล", r"ไพ่พนัน", r"หวยผิดกฎหมาย"
    ],
    "COUNTERFEIT": [
        r"กระเป๋าแบรนด์ปลอม", r"รองเท้า fake", r"software เถื่อน", r" replica",
        r" mirror grade", r"งานก๊อป", r"เกรดมิลเลอร์", r"ของแท้ 100% ปลอม", r"windows แท้ 50"
    ],
    "CONTROLLED_GOODS": [
        r"ยาอันตราย", r"เวชภัณฑ์", r"เครื่องมือแพทย์", r"ยาควบคุม", r"ยาปฏิชีวนะ", r"ยาควบคุมพิเศษ"
    ]
}

EXCLUSION_RULES = [
    r"ปืนฉีดน้ำ", r"ปืนของเล่น", r"เด็กเล่น", r"บีบีกัน.*(ถูกกฎหมาย|ไม่ดัดแปลง)",
    r"กัญชาแมว", r"โมเดล.*ปืน", r"สเกล.*1/", r"หนังสือ.*ยา"
]

def map_category(row):
    """
    ตรวจสอบข้อความ (Title, Description) และ Flags เดิม เพื่อระบุหมวดหมู่ความผิดปกติ
    - อัปเดต: ให้ความสำคัญกับ Ground Truth จากไฟล์ต้นฉบับก่อนเพื่อลดปัญหา Overfitting จาก Regex
    """
    title = str(row.get("Title", ""))
    desc = str(row.get("Description", ""))
    text = f"{title} {desc}".lower()
    
    gt = str(row.get("Ground Truth", "")).strip().upper()
    counterfeit_flag = row.get("Counterfeit_flag", 0)
    illegal_flag = row.get("Illegal Product_flag", 0)
    
    # 1. เช็กตาม Ground Truth ก่อน (Human Annotated / Actual Data)
    if gt and gt != "NAN" and gt != "":
        if gt == "COUNTERFEIT" or counterfeit_flag == 1:
            return "COUNTERFEIT"
        elif gt == "PROHIBITED_PRODUCT" or illegal_flag == 1:
            # พยายามระบุหมวดหมู่ย่อยของ PROHIBITED_PRODUCT
            if "id" in text or "card" in text or "currency" in text or "เอกสาร" in text:
                return "ILLEGAL_DOCS"
            elif "firearm" in text or "handgun" in text or "gun" in text or "ปืน" in text:
                return "WEAPON"
            elif "wildlife" in text or "animal" in text or "สัตว์" in text:
                return "WILDLIFE_ANIMAL"
            return "WEAPON" # Default Prohibited Fallback
        elif gt == "NORMAL":
            return "NORMAL"
        else:
            if gt in PROHIBITED_RULES.keys():
                return gt

    # 2. เช็กกฎยกเว้น (Exclusion Rules) ถ้ามีคำที่ยกเว้น ถือว่าปกติ (ลด False Positive)
    for pattern in EXCLUSION_RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return "NORMAL"
            
    # 3. ถ้าไม่มีข้อมูล Ground Truth ค่อยใช้ Rules & Regex ในการสแกนจับ
    for cat, regex_list in PROHIBITED_RULES.items():
        for pattern in regex_list:
            if re.search(pattern, text, re.IGNORECASE):
                return cat
                
    return "NORMAL"

def assign_risk_and_status(category):
    """
    กำหนดค่าความเสี่ยง (ai_risk_level) และสถานะ (ai_status) สอดคล้องกัน
    - สูง (HIGH)       -> บล็อกแล้ว (BLOCKED)
    - ปานกลาง (MEDIUM) -> รอเจ้าหน้าที่ (PENDING_REVIEW)
    - ต่ำ (LOW)        -> อนุมัติ (APPROVED)
    """
    high_risk_categories = {
        "WEAPON", "DRUG", "ILLEGAL_DOCS", "HAZARDOUS",
        "WILDLIFE_ANIMAL", "GAMBLING"
    }
    medium_risk_categories = {
        "COUNTERFEIT", "CONTROLLED_GOODS"
    }
    
    if category in high_risk_categories:
        return "HIGH", "BLOCKED"
    elif category in medium_risk_categories:
        return "MEDIUM", "PENDING_REVIEW"
    else:
        return "LOW", "APPROVED"

def process_dataset():
    print("=" * 60)
    print("🚀 เริ่มกระบวนการ Data Labeling & Preparation สำหรับ AI Moderation")
    print("=" * 60)
    
    prepared_files = {}
    
    for split in SPLITS:
        input_path = os.path.join(OUTPUT_DIR, f"listing_{split}_clean.csv")
        if not os.path.exists(input_path):
            print(f"⚠️ ไม่พบไฟล์: {input_path}")
            continue
            
        df = pd.read_csv(input_path)
        print(f"\n📦 ประมวลผลชุดข้อมูล: {split} (จำนวน {len(df)} แถว)")
        
        # 1. สร้างหมวดหมู่ AI Category
        df["ai_category"] = df.apply(map_category, axis=1)
        
        # 2. กำหนด Risk Level และ Status
        risk_status_tuple = df["ai_category"].apply(assign_risk_and_status)
        df["ai_risk_level"] = [rs[0] for rs in risk_status_tuple]
        df["ai_status"] = [rs[1] for rs in risk_status_tuple]
        
        # 3. สร้าง Text Column หลักสำหรับเข้าโมเดล
        df["text_full"] = df["Title_clean"].fillna("") + " " + df["Description_clean"].fillna("")
        
        # บันทึกไฟล์ที่เตรียมเรียบร้อยแล้ว
        output_path = os.path.join(OUTPUT_DIR, f"safety_{split}_prepared.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        prepared_files[split] = df
        print(f"✅ บันทึกแล้วที่: {output_path}")
        
    # สรุปสถิติข้อมูล Training Set
    if "train" in prepared_files:
        train_df = prepared_files["train"]
        print("\n" + "=" * 60)
        print("📊 สรุปการกระจายตัวของข้อมูล Training Set (safety_train_prepared.csv)")
        print("=" * 60)
        print("\n1. การกระจายตัวของหมวดหมู่ (ai_category):")
        print(train_df["ai_category"].value_counts())
        
        print("\n2. การกระจายตัวของระดับความเสี่ยง (ai_risk_level):")
        print(train_df["ai_risk_level"].value_counts())
        
        print("\n3. การกระจายตัวของสถานะ (ai_status):")
        print(train_df["ai_status"].value_counts())
        print("=" * 60)

if __name__ == "__main__":
    process_dataset()
