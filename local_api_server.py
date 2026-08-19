import re
import os
import sys
import joblib
import datetime
from typing import List, Optional
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from image_safety_inspector import inspect_multiple_images

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(
    title="2HandToYou Local AI Moderation & Suggestion API",
    description="API สำหรับตรวจจับประกาศขายสินค้าผิดปกติ สินค้าห้ามขาย และระบบช่วยแนะนำคำอธิบายสินค้าแบบ Local Only",
    version="1.1.0"
)

# อนุญาต CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด Model Artifacts
MODEL_DIR = "models"
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "safety_classifier.joblib")

vectorizer = None
model = None

if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
    vectorizer = joblib.load(VEC_PATH)
    model = joblib.load(MODEL_PATH)
    print("✅ โหลดโมเดล Local Safety Model เรียบร้อยแล้ว!")
else:
    print("⚠️ ยังไม่พบโมเดล ให้รัน train_local_safety_model.py ก่อน")

CATEGORY_RISK_MAP = {
    "WEAPON": ("HIGH", "BLOCKED"),
    "DRUG": ("HIGH", "BLOCKED"),
    "ILLEGAL_DOCS": ("HIGH", "BLOCKED"),
    "HAZARDOUS": ("HIGH", "BLOCKED"),
    "WILDLIFE_ANIMAL": ("HIGH", "BLOCKED"),
    "GAMBLING": ("HIGH", "BLOCKED"),
    "COUNTERFEIT": ("MEDIUM", "PENDING_REVIEW"),
    "CONTROLLED_GOODS": ("MEDIUM", "PENDING_REVIEW"),
    "NORMAL": ("LOW", "APPROVED")
}

PROHIBITED_RULES = {
    "WEAPON": [r"ปืน", r"กระสุน", r"มีดผิดกฎหมาย", r"มีดพก", r"สนับมือ", r"ดัดแปลงอาวุธ", r"handgun", r"firearm", r"weapon", r"rifle", r"pistol", r"shotgun", r"ammo"],
    "DRUG": [r"ยาเสพติด", r"ยาบ้า", r"กัญชา", r"ไอซ์", r"บ้อง", r"กระท่อม", r"meth", r"cannabis", r"heroin", r"cocaine"],
    "ILLEGAL_DOCS": [r"ธนบัตรปลอม", r"แบงค์ปลอม", r"บัตรประชาชน", r" fake id", r"counterfeit currency", r"วุฒิการศึกษา", r"ใบขับขี่ปลอม", r"passport"],
    "WILDLIFE_ANIMAL": [r"สัตว์ป่า", r"คุ้มครอง", r"ซากสัตว์", r"wildlife", r"protected species", r"งูพิษ", r"งาช้าง", r"นกขุนทอง"],
    "HAZARDOUS": [r"สารเคมี", r"วัตถุระเบิด", r"พลุ", r"ประทัดรุนแรง", r" explosive", r"chemical", r"acid", r"สารพิษ"],
    "GAMBLING": [r"ตู้สล็อต", r"โพยหวย", r"อุปกรณ์พนัน", r" gambling", r"slot machine", r"ไฮโล", r"หวยผิดกฎหมาย"],
    "COUNTERFEIT": [r"กระเป๋าแบรนด์ปลอม", r"รองเท้า fake", r"software เถื่อน", r" replica", r" mirror grade", r"งานก๊อป", r"เกรดมิลเลอร์"],
    "CONTROLLED_GOODS": [r"ยาอันตราย", r"เวชภัณฑ์", r"เครื่องมือแพทย์", r"ยาควบคุม", r"ยาปฏิชีวนะ"]
}

class InspectionRequest(BaseModel):
    title: str
    description: str
    category_user_selected: Optional[str] = None

class SuggestionRequest(BaseModel):
    title: str
    condition: Optional[str] = "USED"
    user_notes: Optional[str] = None

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "2HandToYou Safety Moderation & Suggestion Local AI Server",
        "version": "1.1.0"
    }

# 🔹 Endpoint 1: ตรวจจับประกาศผิดปกติ (AI Safety Moderation)
@app.post("/api/v1/ai/inspect-listing")
async def inspect_listing(
    title: str = Form(...),
    description: str = Form(...),
    category_user_selected: Optional[str] = Form(None),
    images: Optional[List[UploadFile]] = File(None)
):
    full_text = f"{title} {description}"
    predicted_cat = "NORMAL"
    confidence = 1.0

    # 1. รันกฎ Rule-based Check ล่วงหน้าเพื่อความแม่นยำสูงสุด
    for cat, regex_list in PROHIBITED_RULES.items():
        for pattern in regex_list:
            if re.search(pattern, full_text, re.IGNORECASE):
                predicted_cat = cat
                confidence = 0.99
                break
        if predicted_cat != "NORMAL":
            break

    # 2. ถ้า Rule-based ไม่เจอ และมีโมเดล ให้ใช้ ML Model ทำนาย
    if predicted_cat == "NORMAL" and vectorizer and model:
        text_vec = vectorizer.transform([full_text])
        model_cat = model.predict(text_vec)[0]
        probas = model.predict_proba(text_vec)[0]
        model_conf = float(max(probas))
        if model_cat != "NORMAL":
            predicted_cat = model_cat
            confidence = model_conf

    risk_level, status = CATEGORY_RISK_MAP.get(predicted_cat, ("LOW", "APPROVED"))
    
    reasons = []
    if predicted_cat != "NORMAL":
        reasons.append(f"AI ตรวจพบเนื้อหาเข้าข่ายสินค้าผิดปกติหมวดหมู่ '{predicted_cat}' (Confidence: {confidence*100:.1f}%)")

    # 3. ตรวจสอบไฟล์รูปภาพด้วย Image Safety Inspector
    image_bytes_list = []
    if images:
        for img_file in images:
            content = await img_file.read()
            if content:
                image_bytes_list.append((content, img_file.filename or "image.jpg"))

    if image_bytes_list:
        img_inspection = inspect_multiple_images(image_bytes_list)
        if img_inspection["all_reasons"]:
            reasons.extend(img_inspection["all_reasons"])
            
        # ปรับความเสี่ยงและสถานะตามผลการตรวจรูปภาพ
        risk_weights = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        if risk_weights.get(img_inspection["highest_risk_level"], 1) > risk_weights.get(risk_level, 1):
            risk_level = img_inspection["highest_risk_level"]
            status = img_inspection["final_status"]

    if not reasons:
        reasons.append("ไม่พบความผิดปกติในข้อความและรูปภาพ")
        
    return {
        "success": True,
        "result": {
            "category": predicted_cat,
            "risk_level": risk_level,
            "status": status
        },
        "metadata": {
            "confidence_score": round(confidence, 4),
            "detected_reasons": reasons,
            "images_received_count": len(images) if images else 0,
            "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
    }

# 🔹 Endpoint 2: AI แนะนำคำอธิบายสินค้า (Product Description Assistant)
@app.post("/api/v1/ai/suggest-description")
async def suggest_description(req: SuggestionRequest):
    clean_title = req.title.strip()
    notes = req.user_notes.strip() if req.user_notes else "สภาพดี พร้อมใช้งาน"
    condition_str = req.condition if req.condition else "มือสอง (Used)"

    # ตรวจสอบความปลอดภัยเบื้องต้นของคำที่ผู้ขายป้อน
    is_safe = True
    warning_msg = None
    for cat, regex_list in PROHIBITED_RULES.items():
        for pattern in regex_list:
            if re.search(pattern, f"{clean_title} {notes}", re.IGNORECASE):
                is_safe = False
                warning_msg = f"คำอธิบายเดิมอาจเข้าข่ายสินค้าห้ามขายหมวดหมู่ '{cat}' โปรดตรวจสอบกฎระเบียบแพลตฟอร์ม"
                break
        if not is_safe:
            break

    suggested_title = f"{clean_title} มือสอง สภาพดี"
    suggested_desc = (
        f"ส่งต่อ {clean_title} มือสอง\n"
        f"- สภาพสินค้า: {condition_str}\n"
        f"- รายละเอียดเพิ่มเติม: {notes}\n"
        f"- สภาพสินค้าตรงปกรูปถ่าย พร้อมจัดส่งด่วน หรือนัดรับได้ครับ/ค่ะ\n"
        f"สนใจสอบถามรายละเอียดหรือขอรูปเพิ่มเติมทักแชทได้เลยครับ"
    )

    return {
        "success": True,
        "suggested_title": suggested_title,
        "suggested_description": suggested_desc,
        "safety_check": {
            "is_safe": is_safe,
            "warning": warning_msg
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("local_api_server:app", host="127.0.0.1", port=8000, reload=True)

