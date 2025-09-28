# 💻 Odyssey: دستیار هوش مصنوعی فارسی شخصی‌سازی‌شده با یادگیری مستمر

<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--2_Fa-blue?style=for-the-badge&logo=huggingface" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Learning-Continual_FT-red?style=for-the-badge" alt="Learning Badge"/>
  <img src="https://img.shields.io/badge/Language-Persian_%2F_Farsi-green?style=for-the-badge" alt="Language Badge"/>
</p>

## معرفی پروژه

این پروژه یک **دستیار هوش مصنوعی شخصی فارسی** است که بر پایه مدل **GPT-2 فارسی** ساخته شده و به صورت مستمر با داده‌های مکالمه کاربر و محتوای تخصصی انجمن **JumpLander** آموزش می‌بیند.

این مدل قادر است:
- پاسخ‌دهی به سوالات برنامه‌نویسی به فارسی.
- تولید نمونه کدهای اجرایی در Python و سایر زبان‌ها.
- یادگیری مستمر از مکالمات کاربر و به‌روزرسانی مدل به صورت پویا.
- تعامل امن و راحت با کاربر با دستورات مدیریتی ساده.

لینک گیت‌هاب پروژه: [Osodyssey/odyssey](https://github.com/Osodyssey/odyssey)

## ⚙️ ویژگی‌ها و معماری

- **مدل پایه:** HooshvareLab/gpt2-fa  
- **یادگیری مستمر:** فاین‌تیون پس از هر ۲۰ مکالمه در فایل buffer  
- **منبع تخصص اولیه:** اسکرپ اختصاصی انجمن JumpLander  
- **روش تولید پاسخ:** استفاده از توکنایزر GPT2 و تنظیمات نمونه‌گیری پیشرفته (top-k, top-p, temperature)  
- **توسعه محلی:** پشتیبانی از CUDA/GPU

### جریان کاری مدل

1. بررسی وجود فایل SEED اولیه؛ اگر موجود نبود، اسکرپ محتوای انجمن برای تولید SEED انجام می‌شود.  
2. شروع مکالمه با کاربر و ذخیره پرسش و پاسخ‌ها در BUFFER.  
3. پس از پر شدن BUFFER (۲۰ مکالمه)، ترکیب SEED + BUFFER و فاین‌تیون مدل.  
4. ذخیره مدل جدید در مسیر `persian_gpt2_personal/` و پاکسازی BUFFER.

### ساختار پوشه‌ها

```
odyssey/
│── persian_gpt2_personal/       # مدل فاین‌تیون شده
│── persian_buffer.txt           # بافر مکالمات
│── seed_from_forum.txt          # سید اولیه
│── your_script_name.py          # فایل اصلی اجرا
│── requirements.txt             # وابستگی‌ها
```

## 🛠️ نصب و اجرا

### پیش‌نیازها
- Python 3.8+
- کتابخانه‌ها: `transformers`, `torch`, `beautifulsoup4`, `requests`, `huggingface_hub`

```bash
# ایجاد محیط مجازی
python -m venv venv
source venv/bin/activate  # لینوکس / مک
# venv\Scripts\activate  # ویندوز
pip install -r requirements.txt
```

### اجرای برنامه

```bash
python your_script_name.py
```

### دستورات مدیریتی در چت
| دستور | توضیحات |
|-------|---------|
| `/exit` | خروج ایمن از برنامه |
| `/push username/repo_name` | آپلود مدل فاین‌تیون شده در HuggingFace Hub (نیاز به HF_TOKEN) |

## 🔧 توسعه و فاین‌تیون
- **Buffer:** مکالمات کاربر در `persian_buffer.txt` ذخیره می‌شوند.  
- **Seed:** حاوی شخصیت (persona) و محتوای اسکرپ شده انجمن JumpLander.  
- **Fine-tune:** با ترکیب Seed + Buffer و استفاده از Trainer در Transformers.

## 📌 نکات مهم
- GPU توصیه می‌شود برای سرعت فاین‌تیون.  
- فایل buffer بعد از هر فاین‌تیون پاک می‌شود.  
- امکان بارگذاری مجدد مدل فاین‌تیون شده از پوشه `persian_gpt2_personal/` وجود دارد.

## 📝 مشارکت و لایسنس
- مشارکت‌ها خوش‌آمدید: Pull Request یا Issue در گیت‌هاب  
- پروژه تحت **MIT License** قرار دارد
