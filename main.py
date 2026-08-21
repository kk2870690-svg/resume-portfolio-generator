import json
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from google import genai
from google.genai import types

# Configuration constants
RESUME_FILE = "resume.txt"
TEMPLATE_FILE = "template.html"
CSS_FILE = "style.css"
OUTPUT_FILE = "index.html"
MIN_RESUME_LENGTH = 50


def setup_environment() -> str:
    """Load environment variables and retrieve the Gemini API key."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is missing.")
        print("Please create a '.env' file based on '.env.example' and add your valid Gemini API Key.")
        sys.exit(1)
    return api_key


def read_and_clean_resume(file_path: str) -> str:
    """Read and clean resume content from text file."""
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] Required input file '{file_path}' was not found.")
        print(f"Please create '{file_path}' and add your resume text before running the generator.")
        sys.exit(1)

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to read '{file_path}': {e}")
        sys.exit(1)

    cleaned = re.sub(r"\n\s*\n+", "\n\n", content).strip()

    if len(cleaned) < MIN_RESUME_LENGTH:
        print(f"[ERROR] Input resume in '{file_path}' is empty or too short (less than {MIN_RESUME_LENGTH} characters).")
        print("Please provide a complete resume text.")
        sys.exit(1)

    return cleaned


def create_gemini_prompt(resume_text: str) -> str:
    """Construct a strict prompt directing Gemini to extract facts without hallucination."""
    return f"""
You are an expert resume parser and portfolio text structuring assistant.
Analyze the following resume text and extract all relevant information into a structured JSON object.

CRITICAL INSTRUCTIONS & CONSTRAINTS:
1. Base ALL content STRICTLY on the information explicitly present in the provided resume.
2. DO NOT invent, assume, extrapolate, or hallucinate any skills, jobs, dates, projects, achievements, or contact links.
3. If a section or field is not mentioned in the resume, set its value to an empty string "" or an empty array [].
4. Return ONLY valid JSON matching the exact schema specified below. Do not wrap the JSON in markdown backticks or add introductory/concluding text.

JSON SCHEMA REQUIREMENT:
{{
  "name": "Full Name",
  "headline": "Short professional title or tag line",
  "summary": "Concise professional summary derived from resume",
  "contact": {{
    "email": "Email address or empty",
    "phone": "Phone number or empty",
    "location": "City/Country or empty",
    "linkedin": "LinkedIn profile URL or empty",
    "github": "GitHub profile URL or empty",
    "website": "Personal website or empty",
    "other_links": ["List of other relevant links or empty"]
  }},
  "skills": ["List of technical, tool, and domain skills"],
  "experience": [
    {{
      "role": "Job Title",
      "company": "Company Name",
      "location": "Location",
      "dates": "Employment period e.g. Jan 2022 - Present",
      "responsibilities": ["Key responsibility or accomplishment bullet points"]
    }}
  ],
  "education": [
    {{
      "degree": "Degree/Qualification",
      "institution": "University/School Name",
      "dates": "Dates attended",
      "details": "Honors, major, or pertinent notes"
    }}
  ],
  "projects": [
    {{
      "title": "Project Name",
      "description": "Project summary",
      "technologies": ["List of technologies used"],
      "link": "Project link or empty"
    }}
  ],
  "achievements": [
    {{
      "title": "Award or Certification Name",
      "description": "Description or issuing organization",
      "date": "Date achieved or empty"
    }}
  ]
}}

RESUME TEXT:
---
{resume_text}
---
"""


def call_gemini_api(api_key: str, prompt: str) -> str:
    """
    Dynamically discovers models available to your key and uses the first working one.
    """
    client = genai.Client(api_key=api_key)

    # 1. Fetch available models from Google AI
    try:
        all_models = [m.name for m in client.models.list()]
    except Exception as e:
        print(f"[ERROR] Failed to list models from Gemini API: {e}")
        sys.exit(1)

    # 2. Filter out audio, image, robotics, and embedding preview models
    excluded = ["audio", "tts", "robotics", "translate", "embedding", "computer-use", "imagen", "veo", "aqa"]
    text_candidates = [
        m.replace("models/", "")
        for m in all_models
        if "gemini" in m.lower() and not any(k in m.lower() for k in excluded)
    ]

    # Include remaining models as fallback candidates
    all_gemini = [m.replace("models/", "") for m in all_models if "gemini" in m.lower()]
    for m in all_gemini:
        if m not in text_candidates:
            text_candidates.append(m)

    if not text_candidates:
        text_candidates = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    print(f"[*] Testing {len(text_candidates)} candidate model(s) available for your API key...")

    # 3. Try candidates until one succeeds
    last_error = None
    for model_name in text_candidates:
        print(f"[*] Connecting to model: '{model_name}'...")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            if response.text:
                print(f"[SUCCESS] Connected successfully using '{model_name}'!")
                return response.text
        except Exception as err:
            last_error = err
            continue

    print(f"[ERROR] None of the available models could generate content. Last error: {last_error}")
    sys.exit(1)


def parse_and_clean_json(raw_response: str) -> dict:
    """Parse JSON string safely and sanitize structure."""
    text = raw_response.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object at root.")
        return data
    except json.JSONDecodeError as err:
        print(f"[ERROR] Failed to parse JSON response from Gemini: {err}")
        print("Raw response received:")
        print(raw_response)
        sys.exit(1)


def generate_html_portfolio(portfolio_data: dict, template_path: str, css_path: str, output_path: str) -> None:
    """Render HTML portfolio using Jinja2 template with embedded CSS."""
    tmpl_p = Path(template_path)
    css_p = Path(css_path)

    if not tmpl_p.exists():
        print(f"[ERROR] Template file '{template_path}' missing.")
        sys.exit(1)

    css_content = ""
    if css_p.exists():
        css_content = css_p.read_text(encoding="utf-8")
    else:
        print(f"[WARNING] Styling file '{css_path}' missing. Generating unstyled portfolio.")

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template(template_path)

    rendered_html = template.render(
        data=portfolio_data,
        css_style=css_content
    )

    out_p = Path(output_path)
    out_p.write_text(rendered_html, encoding="utf-8")
    print(f"[SUCCESS] Portfolio generated successfully: {out_p.resolve()}")


def main():
    """Main execution flow."""
    print("==================================================")
    print("      AI Resume-to-Portfolio Generator")
    print("==================================================")

    # 1. Check API Key
    api_key = setup_environment()

    # 2. Read and clean resume
    print(f"[*] Reading and cleaning '{RESUME_FILE}'...")
    cleaned_resume = read_and_clean_resume(RESUME_FILE)

    # 3. Construct prompt
    print("[*] Preparing prompt for Gemini API...")
    prompt = create_gemini_prompt(cleaned_resume)

    # 4. Call Gemini API with Auto-Model Selection
    print("[*] Contacting Gemini API...")
    raw_json = call_gemini_api(api_key, prompt)

    # 5. Parse JSON
    print("[*] Parsing and validating structured JSON...")
    portfolio_data = parse_and_clean_json(raw_json)

    # 6. Generate Portfolio Webpage
    print(f"[*] Rendering HTML portfolio to '{OUTPUT_FILE}'...")
    generate_html_portfolio(portfolio_data, TEMPLATE_FILE, CSS_FILE, OUTPUT_FILE)

    print("\nProcess finished! Open 'portfolio.html' in your browser to view your webpage.")


if __name__ == "__main__":
    main()