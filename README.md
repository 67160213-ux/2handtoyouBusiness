# 2HandToYou Business — Local AI Safety Moderation & Recommendation System

ระบบบริการ **Local AI Safety Moderation & Product Description Assistant** สำหรับแพลตฟอร์มซื้อขายสินค้ามือสองออนไลน์ ประมวลผลแบบ Local Only รวดเร็ว ปลอดภัย และไม่เสียค่าใช้จ่าย API ภายนอก

---

## 📌 ภาพรวมโครงการ (Overview)

โปรเจกต์นี้ประกอบด้วย 2 ระบบหลัก:
1. **Abnormal Listing Detection & Safety Moderation:** ตรวจจับประกาศขายสินค้าผิดปกติ สินค้าละเมิดกฎหมาย สินค้าห้ามขาย (อาวุธ, ยาเสพติด, สินค้าละเมิดลิขสิทธิ์ ฯลฯ) พร้อมประเมินค่าความเสี่ยงและกำหนดสถานะการดำเนินการ (BLOCKED / PENDING_REVIEW / APPROVED)
2. **Product Description Assistant:** ระบบช่วยเหลือผู้ขายในการเรียบเรียงและปรับปรุงคำอธิบายสินค้าให้น่าซื้อ ครบถ้วน และถูกต้องตามกฎระเบียบแพลตฟอร์ม

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
2handtoyouBusiness/
├── data/                            # โฟลเดอร์เก็บไฟล์ข้อมูลดิบ (.xlsx)
│   ├── dataset_abnormal_listing.xlsx
│   └── dataset_recommendation.xlsx
├── output/                          # โฟลเดอร์เก็บไฟล์ CSV ที่ผ่านการทำความสะอาดแล้ว
│   ├── listing_train_clean.csv / _val_ / _test_ / _uat_
│   ├── safety_train_prepared.csv
│   └── recommendation_clean.csv
├── models/                          # โฟลเดอร์เก็บ Model Artifacts ที่ฝึกสอนแล้ว
│   ├── tfidf_vectorizer.joblib
│   └── safety_classifier.joblib
├── data_cleaning_pipeline.py        # สคริปต์ Data Cleaning & Feature Engineering
├── prepare_safety_dataset.py        # สคริปต์ทำ Data Labeling 8 หมวดหมู่ความผิดปกติ
├── train_local_safety_model.py      # สคริปต์ฝึกสอน Local Model (TF-IDF + Classifier)
├── image_safety_inspector.py        # โมดูลวิเคราะห์ความปลอดภัยรูปภาพ (PIL Based)
├── local_api_server.py              # FastAPI REST Server (Port 8000)
├── test_api.py                      # สคริปต์ยิงทดสอบ API Endpoints
├── Dockerfile                       # Docker Container Definition
├── docker-compose.yml               # Docker Compose Orchestration Setup
├── requirements.txt                 # รายการ Python Packages
└── README.md                        # เอกสารอธิบายโครงการและการใช้งาน
```

---

## 🐳 Docker Deployment Guide (สำหรับทีม SE / Backend)

### 📌 ข้อมูลสำคัญ Container
* **Container Name:** `2handtoyou_ai_safety_api`
* **Exposed Port:** `8000`
* **Base Image:** `python:3.11-slim`

### 🚀 การสั่งรันด้วย Docker Compose (แนะนำ)

1. เปิด Terminal ในโฟลเดอร์โปรเจกต์
2. สั่ง Build และ Start Container ในฉากหลัง:
   ```bash
   docker compose up -d --build
   ```
3. ตรวจสอบสถานะการทำงาน:
   ```bash
   docker compose ps
   ```
4. ดู Log การทำงานของ AI Server:
   ```bash
   docker compose logs -f ai-moderation-service
   ```
5. สั่งหยุดการทำงาน:
   ```bash
   docker compose down
   ```

---

### 🚀 การสั่งรันด้วย Docker CLI โดยตรง

1. **Build Image:**
   ```bash
   docker build -t 2handtoyou-ai-api:1.0 .
   ```

2. **Run Container:**
   ```bash
   docker run -d \
     --name 2handtoyou_ai_safety_api \
     -p 8000:8000 \
     --restart always \
     2handtoyou-ai-api:1.0
   ```

---

## 🧪 การทดสอบยิง API (API Endpoints & Testing)

### 1. Health Check Endpoint
```bash
curl http://localhost:8000/
```

### 2. Test Inspect Listing Endpoint (`POST /api/v1/ai/inspect-listing`)
```bash
curl -X POST http://localhost:8000/api/v1/ai/inspect-listing \
  -F "title=ขายปืนสั้นมือสอง" \
  -F "description=แถมกระสุนซ้อม 50 นัด"
```

**Response Payload:**
```json
{
  "success": true,
  "result": {
    "category": "WEAPON",
    "risk_level": "HIGH",
    "status": "BLOCKED"
  },
  "metadata": {
    "confidence_score": 0.99,
    "detected_reasons": [
      "AI ตรวจพบเนื้อหาเข้าข่ายสินค้าผิดปกติหมวดหมู่ 'WEAPON' (Confidence: 99.0%)"
    ],
    "images_received_count": 0
  }
}
```

### 3. Test Suggest Description Endpoint (`POST /api/v1/ai/suggest-description`)
```bash
curl -X POST http://localhost:8000/api/v1/ai/suggest-description \
  -H "Content-Type: application/json" \
  -d '{"title": "ขายรองเท้า Nike Air Jordan 1", "condition": "USED_LIKE_NEW", "user_notes": "ใส่น้อยครั้ง มีกล่องครบ Size 42"}'
```

---

## 💻 การรันโปรแกรมในโหมดการพัฒนา (Development & Pipeline Execution)

```bash
# 1. ทำความสะอาดข้อมูล
python data_cleaning_pipeline.py

# 2. ทำ Data Labeling สำหรับ Safety Model
python prepare_safety_dataset.py

# 3. ฝึกสอนโมเดล Local Safety Model
python train_local_safety_model.py

# 4. เปิดรัน API Server แบบ Local
python local_api_server.py

# 5. ทดสอบยิง API ในอีก Terminal
python test_api.py
```