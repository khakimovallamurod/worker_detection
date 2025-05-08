````markdown
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
2. Telegram'dan `chat_id` ni aniqlang (buni olishda yordam kerak bo‘lsa, so‘rashingiz mumkin).

### Dasturni ishga tushirish:

```python
import asyncio
asyncio.run(main("test_video.mp4", "output.mp4", "YOUR_BOT_TOKEN", "YOUR_CHAT_ID"))
```

---

## 📎 Eslatma

* `Fire` yoki `Smoke` birinchi marta aniqlanganda `Tracker ID` yordamida faqat bir marta xabar yuboriladi.
* Video tugaguncha yoki `q` tugmasi bosilmaguncha oynada real vaqtli monitoring davom etadi.

---
