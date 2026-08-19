# -*- coding: utf-8 -*-
"""
train_local_safety_model.py
-------------------------------------------------------------------
สคริปต์ฝึกสอน Baseline Local AI Model สำหรับตรวจจับประกาศผิดปกติ
- ฝึกสอน Text Classification Model (TF-IDF + LogisticRegression / SGDClassifier)
- ทำนาย 3 ค่าเป้าหมาย: ai_category, ai_risk_level, ai_status
- บันทึกโมเดลไว้ในโฟลเดอร์ models/ เพื่อนำไปใช้งานกับ FastAPI Server
-------------------------------------------------------------------
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# บังคับ unbuffered print
def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# เส้นทางไฟล์
TRAIN_PATH = "output/safety_train_prepared.csv"
VAL_PATH   = "output/safety_val_prepared.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Mapping Rule สำหรับแปลง Category เป็น Risk Level และ Status
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

def train_model():
    log("=" * 60)
    log("🤖 เริ่มการฝึกสอน Baseline Local Safety Model")
    log("=" * 60)
    
    # 1. โหลดข้อมูล
    train_df = pd.read_csv(TRAIN_PATH)
    val_df   = pd.read_csv(VAL_PATH)
    
    X_train_text = train_df["text_full"].fillna("")
    y_train      = train_df["ai_category"]
    
    X_val_text   = val_df["text_full"].fillna("")
    y_val        = val_df["ai_category"]
    
    log(f"📦 Train set: {len(X_train_text)} รายการ | Validation set: {len(X_val_text)} รายการ")
    
    # 2. Vectorization (TF-IDF)
    log("\n🔍 กำลังแปลงข้อความด้วย TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_val_vec   = vectorizer.transform(X_val_text)
    
    # 3. Model Training
    log("🏋️ กำลังฝึกสอน Logistic Regression Classifier (class_weight='balanced')...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=1.0,
        random_state=42
    )
    model.fit(X_train_vec, y_train)
    
    # 4. Evaluation
    y_pred = model.predict(X_val_vec)
    y_proba = model.predict_proba(X_val_vec).max(axis=1)
    
    acc = accuracy_score(y_val, y_pred)
    log(f"\n✅ Validation Accuracy: {acc * 100:.2f}%")
    log("\n📊 Classification Report (หมวดหมู่ความผิดปกติ):")
    log(classification_report(y_val, y_pred, zero_division=0))
    
    # 5. ประเมินผล Risk Level และ Status
    val_df["pred_category"] = y_pred
    val_df["pred_confidence"] = y_proba
    
    risk_status = val_df["pred_category"].map(lambda c: CATEGORY_RISK_MAP.get(c, ("LOW", "APPROVED")))
    val_df["pred_risk_level"] = [rs[0] for rs in risk_status]
    val_df["pred_status"]     = [rs[1] for rs in risk_status]
    
    status_acc = accuracy_score(val_df["ai_status"], val_df["pred_status"])
    log(f"🎯 Action Status Accuracy (BLOCKED / PENDING_REVIEW / APPROVED): {status_acc * 100:.2f}%")
    
    # 6. บันทึก Model Artifacts
    vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODEL_DIR, "safety_classifier.joblib")
    
    joblib.dump(vectorizer, vec_path)
    joblib.dump(model, model_path)
    
    log("\n" + "=" * 60)
    log(f"💾 บันทึกโมเดลเรียบร้อยแล้วที่:")
    log(f"   - Vectorizer: {vec_path}")
    log(f"   - Model:      {model_path}")
    log("=" * 60)

if __name__ == "__main__":
    train_model()
