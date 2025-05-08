# 🏗️ Qurilish maydonini kuzatuvchi tizim (YOLO + Telegram ogohlantirishlar)

Bu loyiha **real vaqtli sun’iy intellekt asosida ishlovchi kuzatuv tizimi** bo‘lib, quyidagilarni amalga oshiradi:

- `Person`, `Helmet`, `Vest` obyektlarini aniqlaydi
- `Fire` va `Smoke` klasslarini aniqlaydi
- Qurilish hududidagi `Person` larni sanaydi
- Har bir obyektni unikal `tracker ID` bilan kuzatadi
- `Fire` yoki `Smoke` aniqlansa, **Telegram** bot orqali xabar yuboradi (annotatsiyalangan rasm bilan)

---

## 📌 Asosiy imkoniyatlar

✅ Real vaqtli video yoki fayldan kadrlar bilan ishlash  
✅ 3 ta mustaqil YOLOv8 modeli yordamida obyektlarni aniqlash  
✅ Har bir obyektni kuzatish (`tracker ID` bilan)  
✅ `Fire` yoki `Smoke` klasslari aniqlanganda Telegram xabarnoma yuborish  
✅ Annotatsiyalangan video faylni saqlash (MP4 formatda)  
✅ Ekranda to‘liq ko‘rinishda oynani ochish va odamlar sonini ko‘rsatish  

---

## 🧠 Ishlatilgan modellar

| Model nomi          | Tavsifi                      | Fayl yo‘li                      |
|---------------------|------------------------------|---------------------------------|
| `person20k.pt`      | `Person` klassini aniqlaydi  | `models/person20k.pt`          |
| `build25k.pt`       | `Helmet`, `Vest` klasslari   | `models/build25k.pt`           |
| `fire-smoke-model.pt` | `Fire`, `Smoke` klasslari    | `models/fire-smoke-model.pt`   |

---

## ⚙️ Dastur qanday ishlaydi?

1. Modellar yuklanadi va video (kamera yoki fayl) ochiladi.
2. Har bir kadrda 3 ta model alohida ishlaydi va aniqlangan obyektlar birlashtiriladi.
3. NMS (Non-Maximum Suppression) orqali ortiqcha qamrovlar olib tashlanadi.
4. Tracker yordamida obyektlar aniqlanadi va kuzatib boriladi.
5. `Fire` yoki `Smoke` topilsa, Telegram bot orqali xabar yuboriladi.
6. Annotatsiyalangan kadrlar ekranga chiqariladi va saqlanadi.

---

## 🛠️ Ishga tushirish

```bash
pip install -r requirements.txt
````

### Telegram bot sozlamalari:

1. [@BotFather](https://t.me/BotFather) orqali bot yarating va `bot_token` oling.
2. Telegram'dan `chat_id` ni aniqlang.

### Dasturni ishga tushirish:

```python
import asyncio
from camera-tracking import main  

bot_token = 'YOUR_TELEGRAM_BOT_TOKEN'
chat_id = 'YOUR_TELEGRAM_CHAT_ID'
video_path = 'input_video.mp4'
output_path = 'output_annotated.mp4'

asyncio.run(main(video_path, output_path, bot_token, chat_id))
```

---

## 📎 Eslatma

* `Fire` yoki `Smoke` birinchi marta aniqlanganda `Tracker ID` yordamida faqat bir marta xabar yuboriladi.
* Video tugaguncha yoki `q` tugmasi bosilmaguncha oynada real vaqtli monitoring davom etadi.

---

---

## 🧪 Testing with Webcam (Optional)

To test with a real-time camera feed, replace this line:

```python
cap = cv2.VideoCapture(video_path)
```

with:

```python
cap = cv2.VideoCapture(0)  # or other camera index
```

---

---

## 🛠 Requirements

* Python 3.8+
* OpenCV
* Ultralytics (YOLOv8)
* Supervision
* Python-Telegram-Bot

---

