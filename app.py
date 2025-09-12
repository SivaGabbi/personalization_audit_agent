import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import google.generativeai as genai
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# --- Setup (Unchanged) ---
kb_df = pd.read_csv('personalization_kb.csv')
vendors_df = pd.read_csv('vendors.csv')

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

app = Flask(__name__)
CORS(app)

# --- Presentation Generation Logic (Modified to Save to File) ---
def create_and_save_presentation(json_data, filename):
    """Generates a PowerPoint presentation and saves it to a local file."""
    prs = Presentation()
    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title, subtitle = slide.shapes.title, slide.placeholders[1]
    url = json_data.get("url", "Website")
    title.text = "Personalization Audit Report"
    subtitle.text = f"An analysis of {url}\nGenerated on {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    
    # Slide 2: Summary
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Executive Summary & Key Findings"
    score = json_data.get("overall_score", 0)
    shape = slide.shapes.add_shape(MSO_SHAPE.DONUT, Inches(0.5), Inches(1.8), Inches(2), Inches(2))
    shape.text = str(score)
    font = shape.text_frame.paragraphs[0].font
    font.size, font.bold, font.color.rgb = Pt(44), True, RGBColor(0, 0, 0)
    content_shape = slide.placeholders[1]
    content_shape.left, content_shape.width = Inches(3.0), Inches(6.5)
    tf = content_shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text, p.font.bold, p.font.size = "Key Findings:", True, Pt(20)
    for finding in json_data.get("key_findings", []):
        p = tf.add_paragraph()
        p.text, p.level = finding, 1
        
    # Recommendation Slides
    for i, rec in enumerate(json_data.get("recommendations", []), 1):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"Recommendation #{i}"
        tf = slide.placeholders[1].text_frame
        tf.text = rec
        tf.paragraphs[0].font.size = Pt(18)
        
    # Save to the specified filename
    prs.save(filename)
    print(f"✅ Presentation saved locally as '{filename}'")

# --- Core Auditing Logic (Unchanged, except for adding URL to result) ---
# [generate_audit_with_llm and get_rag_context remain here]
def get_rag_context(findings):
    if not findings: return "No specific personalization keywords found on page."
    # ... (rest of function is unchanged)
    findings_text = " ".join(findings).lower()
    context = []
    for _, row in kb_df.iterrows():
        keywords = row['evidence_keywords'].split(',')
        if any(keyword.strip() in findings_text for keyword in keywords):
            context.append(f"- Use Case: {row['use_case']}\n  Best Practice: {row['best_practice']}")
    return "\n".join(context) if context else "No matching best practices found."

def generate_audit_with_llm(url, content_findings, tech_findings):
    if not model: raise ConnectionError("Gemini model is not initialized.")
    # ... (prompt is unchanged)
    rag_context = get_rag_context(content_findings)
    prompt = f"""
    You are a world-class Personalization and CRO (Conversion Rate Optimization) expert AI. Your task is to analyze evidence from a website to produce an insightful audit report.
    **Website Audited:** {url}
    **Evidence from Page Content (Keywords):**
    ```
    {json.dumps(content_findings, indent=2)}
    ```
    **Detected Third-Party Technologies (A/B Testing, Personalization, CDPs):**
    ```
    {json.dumps(tech_findings, indent=2)}
    ```
    **Relevant Personalization Best Practices (for context):**
    ```
    {rag_context}
    ```
    **Instructions:**
    Analyze all evidence to generate a JSON audit report. Your entire output must be a single, valid JSON object with three keys: "overall_score" (integer 0-100), "key_findings" (list of strings), and "recommendations" (list of strings).
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        result_data = json.loads(cleaned_response)
        result_data['url'] = url  # Ensure URL is in the data for the presentation
        return result_data
    except Exception as e:
        print(f"Error generating content from LLM: {e}")
        return {"error": "Failed to get analysis from AI model."}

# --- Main Audit Endpoint (Modified to auto-save presentation) ---
@app.route('/audit', methods=['POST'])
def audit_website():
    url = request.json.get('url')
    if not url: return jsonify({"error": "URL is required"}), 400
    
    try:
        # --- The entire agent scraping logic is unchanged ---
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        # ... (rest of Selenium setup)
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        page_source = driver.page_source.lower()
        content_findings = []
        all_keywords = kb_df['evidence_keywords'].str.split(',').explode().unique()
        for keyword in all_keywords:
            k = keyword.strip()
            if k in page_source: content_findings.append(f"Found keyword evidence: '{k}'")
        tech_findings = []
        scripts = driver.execute_script("return Array.from(document.scripts).map(s => s.src);")
        script_sources_text = " ".join(filter(None, scripts))
        for _, vendor in vendors_df.iterrows():
            for signature in vendor['signatures'].split(','):
                is_present = False
                if signature in script_sources_text: is_present = True
                else:
                    try:
                        if driver.execute_script(f"return typeof {signature} !== 'undefined'"): is_present = True
                    except: pass
                if is_present:
                    tech_findings.append(f"Detected {vendor['vendor_name']} ({vendor['category']})")
                    break
        driver.quit()
        if not content_findings: content_findings.append("No basic personalization keywords detected.")
        if not tech_findings: tech_findings.append("No specialized third-party personalization or A/B testing tools were detected.")
        
        # --- Generate analysis ---
        analysis_result = generate_audit_with_llm(url, list(set(content_findings)), list(set(tech_findings)))
        
        # --- NEW: Automatically create and save the presentation ---
        if 'error' not in analysis_result:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # Sanitize URL for filename
            safe_url = url.replace('https://','').replace('http://','').replace('/','_').split('.')[0]
            filename = f"Audit_Report_{safe_url}_{timestamp}.pptx"
            create_and_save_presentation(analysis_result, filename)
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({"error": f"An error occurred in the backend: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)