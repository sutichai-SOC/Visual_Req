# VisualReq — คู่มือติดตั้ง & ใช้งาน

เครื่องมือช่วยงาน QA: อ่าน requirement, Gen Test Case / Bug Card ด้วย AI, และ **ลงเวลา (Log Time) ใน Jira** พร้อมดู Timesheet แบบปฏิทิน

ตัวแอปมี 2 ไฟล์เท่านั้น ทำงานบนเครื่องตัวเอง (localhost) ไม่ต้องติดตั้งอะไรเพิ่มนอกจาก Python

---

## สิ่งที่ต้องมี

- **Python 3** (เวอร์ชัน 3.8 ขึ้นไป) — macOS มักมีมาให้อยู่แล้ว
  - เช็กว่ามีไหม เปิด Terminal พิมพ์: `python3 --version`
  - ถ้ายังไม่มี โหลดได้ที่ https://www.python.org/downloads/
- อินเทอร์เน็ต (แอปโหลด React/Tailwind จาก CDN และเรียก Jira / AI ผ่านเน็ต)
- ไม่ต้อง `pip install` อะไรเลย ใช้ไลบรารีมาตรฐานของ Python ล้วน ๆ

---

## วิธีติดตั้ง / รัน

1. แตกไฟล์ zip นี้ออกมา จะได้โฟลเดอร์ `VisualReq` ที่มี 2 ไฟล์:
   - `server.py`
   - `visualreq_web_app_v2.html`

2. เปิด **Terminal** แล้ว `cd` เข้าไปในโฟลเดอร์นั้น เช่น:
   ```bash
   cd ~/Downloads/VisualReq
   ```

3. สั่งรัน server:
   ```bash
   python3 server.py
   ```
   จะเห็นข้อความประมาณนี้:
   ```
   VisualReq running →  http://127.0.0.1:5173/visualreq_web_app_v2.html
   Press Ctrl+C to stop.
   ```

4. เปิดเบราว์เซอร์ (แนะนำ Chrome) ไปที่:
   ```
   http://localhost:5173/visualreq_web_app_v2.html
   ```

5. เลิกใช้งาน กลับไปที่ Terminal กด **Ctrl + C** เพื่อปิด server

---

## ตั้งค่าครั้งแรก (ในแอป)

กดไปที่หน้า **ตั้งค่า / Settings** ในแอป แล้วใส่ค่าเหล่านี้ (เก็บไว้ในเบราว์เซอร์ของคุณเองเท่านั้น ไม่ได้ส่งไปไหน):

**สำหรับ AI (Gen Test Case / Bug Card):**
- เลือกผู้ให้บริการ (Claude / OpenAI / Gemini) และใส่ **API Key** ของตัวเอง

**สำหรับ Jira (Log Time / รายงาน / สร้าง Subtask):**
- **Base URL** เช่น `https://your-company.atlassian.net`
- **Email** อีเมลที่ใช้ล็อกอิน Jira
- **API Token** สร้างได้ที่ https://id.atlassian.com/manage-profile/security/api-tokens

> 🔑 API Key / Token ทั้งหมดเก็บอยู่ใน localStorage ของเบราว์เซอร์คุณเอง — **ไม่ได้ฝังมากับไฟล์ที่แชร์** ใครเอาแอปไปรันก็ต้องใส่ของตัวเอง

---

## ปัญหาที่เจอบ่อย

**รันแล้วขึ้น `Address already in use` (พอร์ต 5173 ถูกใช้อยู่)**
มี server เก่าค้างอยู่ ปิดทิ้งก่อนแล้วรันใหม่:
```bash
lsof -ti tcp:5173 | xargs kill
python3 server.py
```

**อยากใช้พอร์ตอื่น** (เช่น 5173 ชนกับโปรแกรมอื่น):
```bash
PORT=8080 python3 server.py
```
แล้วเปิด `http://localhost:8080/visualreq_web_app_v2.html`

**หน้าเว็บเปิดไม่ขึ้น / ว่างเปล่า**
- เช็กว่า Terminal ที่รัน server ยังเปิดอยู่ (ปิด Terminal = ปิด server)
- เช็กว่าต่ออินเทอร์เน็ตอยู่ (แอปโหลด React/Tailwind จาก CDN)

**Windows:** ใช้ `python` แทน `python3` และปิดพอร์ตด้วย `netstat -ano | findstr :5173` แล้ว `taskkill /PID <เลข> /F`

---

## หมายเหตุ
- แอปทำงานบนเครื่องตัวเอง ข้อมูลไม่ได้ขึ้น server กลางที่ไหน
- `server.py` เป็นแค่ตัวกลาง (proxy) คุยกับ Jira / AI ให้ เพราะเบราว์เซอร์เรียกตรงไม่ได้ (ติด CORS)

---

## Deploy ขึ้นเว็บ (เช่น Render) — ให้ทีมใช้ร่วมกัน

1. เอา `server.py`, `visualreq_web_app_v2.html`, `README.md` ขึ้น GitHub (แนะนำ repo แบบ Private)
2. Render → **New → Web Service** → เลือก repo แล้วตั้งค่า:
   - Build Command: `echo ok`
   - Start Command: `python3 server.py`
   - Instance type: Free
3. **Environment Variables**
   | ชื่อ | ค่า | จำเป็น |
   | --- | --- | --- |
   | `HOST` | `0.0.0.0` | ✅ |
   | `APP_PASSWORD` | รหัสผ่านเข้าแอป (ตั้งให้เดายาก) | ✅ |
   | `APP_USER` | ชื่อผู้ใช้ (ไม่ใส่ = ใส่ชื่ออะไรก็ได้) | ไม่บังคับ |
   | `JIRA_ALLOWED_HOSTS` | host Jira อื่นนอกจาก `*.atlassian.net` คั่นด้วย comma | ไม่บังคับ |
   | `AI_ALLOWED_HOSTS` | host AI อื่นนอกจาก OpenAI / Azure OpenAI | ไม่บังคับ |
4. เปิด `https://<ชื่อแอป>.onrender.com/` → เบราว์เซอร์จะถามชื่อผู้ใช้/รหัสผ่าน
5. Health check path (ถ้าแพลตฟอร์มถาม): `/healthz`

## Security (ที่ server.py ทำให้)
- มีรหัสผ่านเข้าแอปเมื่อตั้ง `APP_PASSWORD` (รันบนเครื่องตัวเองไม่ต้องตั้ง)
- เรียก Jira ได้เฉพาะ `https://*.atlassian.net` และ AI ได้เฉพาะ Anthropic / OpenAI / Azure OpenAI / Gemini — กันไม่ให้ server ถูกใช้เป็นทางผ่านไปเว็บอื่น
- บน server ตรวจใบรับรอง HTTPS เสมอ (การข้ามการตรวจใช้ได้เฉพาะตอนรันบน localhost)
- เปิดให้โหลดเฉพาะไฟล์หน้าแอป ไฟล์อื่นในโฟลเดอร์ (รวม `server.py`) โหลดไม่ได้
- จำกัดขนาด request ที่ 5 MB
- API Key / Jira Token ยังเก็บใน localStorage ของแต่ละคน — ไม่อยู่บน server
# Visual_Req
