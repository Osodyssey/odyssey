# 💻 JumpLander Personal odyssey: دستیار هوش مصنوعی فارسی با یادگیری مستمر

<p align="center">
  <img src="https://img.shields.io/badge/Model-GPT--2_Fa-blue?style=for-the-badge&logo=huggingface" alt="Model Badge"/>
  <img src="https://img.shields.io/badge/Learning-Continual_FT-red?style=for-the-badge" alt="Learning Badge"/>
  <img src="https://img.shields.io/badge/Language-Persian_%2F_Farsi-green?style=for-the-badge" alt="Language Badge"/>
</p>

## معرفی پروژه و نویسنده

این پروژه یک **سامانه خودنوشت** برای ساخت یک **دستیار هوش مصنوعی شخصی‌سازی‌شده و متخصص فارسی** است. هسته اصلی این دستیار، مدل زبان بزرگ (LLM) **GPT-2 فارسی** است که توسط خودم، **JumpLander / Osodyssey**، توسعه یافته تا دانش و تخصص را از طریق یک فرآیند **یادگیری مستمر (Continual Learning)** کسب کند.

هدف اصلی این پروژه، غلبه بر محدودیت‌های مدل‌های عمومی و ایجاد یک مدل تخصصی است که نه تنها به فارسی پاسخ می‌دهد، بلکه با دانش و سبک نوشتاری مرتبط با پروژه‌ها و حوزه‌های فنی من آشنایی کامل دارد.

## 💡 معماری و نحوه کار

معماری این مدل بر پایه چرخه **Seed and Buffer Fine-tuning** است که از فراموشی فاجعه‌بار جلوگیری کرده و به طور موثر مدل را در طول زمان به روز نگه می‌دارد.

### نمودار مقایسه مدل پایه و مدل شخصی‌سازی شده

| ویژگی | مدل GPT-2 فارسی عمومی (Base) | مدل شخصی JumpLander (Fine-Tuned) |
| :--- | :--- | :--- |
| **پایگاه دانش** | کورپوس عمومی و ایستا اینترنت | دانش عمومی + **محتوای تخصصی و مکالمات کاربر** |
| **روش یادگیری** | استاتیک، فقط آموزش اولیه | **پویا، فاین‌تیون مستمر (Continual Learning)** |
| **شخصیت (Persona)** | خنثی و عمومی | **متخصص برنامه‌نویسی فارسی** (با تمرکز بر سبک و اصطلاحات JumpLander) |
| **منبع تخصص اولیه** | ❌ ندارد | ✅ **اسکرپ هدفمند از JumpLander Forum Index** |

### جریان کاری مدل (Conceptual Flow)

```mermaid
graph TD
    A[GPT-2 Base Model] --> B(بارگذاری);
    B --> C{آیا SEED_FILE موجود است؟};
    C -- بله --> D[بارگذاری SEED (Persona + Forum Data)];
    C -- خیر --> E[ساخت SEED از Persona و https://jumplander.org/?fa=forum];
    E --> D;
    D --> F(شروع چت);
    F --> G[تعامل کاربر/مدل];
    G --> H[ذخیره در BUFFER (persian_buffer.txt)];
    H -- بافر کامل شد (20 مکالمه) --> I[ترکیب SEED + BUFFER];
    I --> J(فاین‌تیون مجدد مدل);
    J --> K[ذخیره مدل جدید در persian_gpt2_personal/];
    K --> F;
```

## 🛠️ ویژگی‌های فنی

- Scraping هدفمند: تابع اختصاصی برای اسکرپ محتوای تخصصی از صفحه فهرست انجمن JumpLander جهت تزریق دانش اولیه به مدل.
- Data Collator سفارشی: استفاده از DataCollatorForLanguageModeling با تنظیمات Causal LM برای آموزش زبان.
- GPU Acceleration: پشتیبانی کامل از CUDA/GPU با استفاده از torch و accelerate برای تسریع فرآیند فاین‌تیونینگ.
- ماژولار بودن: ساختار تمیز توابع برای مدیریت آسان‌تر مراحل آموزش، تولید و ذخیره‌سازی.

## 🚀 راهنمای نصب و اجرا

### ۱. پیش‌نیازها
برای اجرای این پروژه، نیاز به Python 3.8+ و کتابخانه‌های زیر دارید:

```bash
# ایجاد محیط مجازی (پیشنهاد شدید)
python -m venv venv
source venv/bin/activate  # لینوکس/مک‌او‌اس
# venv\Scriptsctivate   # ویندوز
```

### ۲. نصب نیازمندی‌ها
تمام نیازمندی‌ها در فایل requirements.txt فهرست شده‌اند:

```bash
pip install -r requirements.txt
```

نکته CUDA: اگر از GPU استفاده می‌کنید، مطمئن شوید که torch را متناسب با نسخه CUDA سیستم خود نصب کرده‌اید.

### ۳. اجرای برنامه

```bash
python your_script_name.py # نام فایل پایتون اصلی را جایگزین کنید
```

## ⌨️ دستورات چت و مدیریت

| دستور | توضیحات |
| :--- | :--- |
| `/exit` | خروج ایمن از برنامه. |
| `/push username/repo_name` | آپلود مدل فاین‌تیون شده در Hugging Face Hub. نیاز به تنظیم متغیر محیطی HF_TOKEN دارد. |
