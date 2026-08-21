# AI-Assisted Resume to Portfolio Generator

An end-to-end Python pipeline that parses raw resume text into structured data using Google Gemini API and renders a responsive, retro-futuristic personal web portfolio using Jinja2 and custom CSS.

---

## 📌 Project Overview

This project automates the creation of professional web portfolios from unstructured resume text. It leverages large language models (LLM) for precise schema extraction and dynamic templating engines to generate accessible, performant, and beautifully styled static web pages.

### Key Features
- **Dynamic Model Discovery:** Automatically queries and fallbacks to active Gemini models (e.g., `gemini-flash-latest`, `gemini-flash-lite-latest`) to avoid deprecated endpoint errors.
- **Strict JSON Schema Enforcement:** Uses structured prompt engineering and regex sanitization to ensure deterministic LLM extraction without hallucinations.
- **Defensive Error Handling:** Validates file availability, token counts, and input thresholds before invoking API calls.
- **Responsive Retro-Futuristic UI:** Styled using an editorial dark theme featuring glassmorphism, gold accents, and subtle cybernetic glowing borders.

---

## 🛠️ Project Structure

```text
resume-portfolio-generator/
│
├── main.py              # Main execution pipeline (parsing, Gemini API, Jinja2 rendering)
├── template.html        # Jinja2 HTML layout template
├── style.css            # Retro-futuristic stylesheet
├── resume.txt           # Sample input resume plain text
├── portfolio.html       # Generated output web portfolio
├── requirements.txt     # Python dependencies
├── .env.example         # Example environment variables template
├── .gitignore           # Git ignore rules for virtual environments and secrets
└── README.md            # Project documentation and architecture details