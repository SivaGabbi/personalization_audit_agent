import os
import time
import uuid
import threading
import json
import pandas as pd
from urllib.parse import urlparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image, ImageChops
import google.generativeai as genai
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# --- In-memory storage for tasks and knowledge base ---
tasks = {}
kb_df = None

# --- Configuration ---
VENDORS_FILE = 'vendors.csv'
KB_FILE = 'personalization_kb.csv'
RECOMMENDATION_KEYWORDS = [
    'recommended for you', 'inspired by your browsing', 'recently viewed',
    'customers also bought', 'frequently bought together', 'similar products',
    'you might also like', 'because you viewed', 'complete your purchase'
]

# --- RAG/LLM Setup ---
llm_model = None
embedding_model = None
try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    embedding_model = 'models/embedding-001'
except Exception as e:
    print(f"⚠️ RAG/LLM Warning: {e}")

# --- NEW: Function to embed the knowledge base on startup ---
def initialize_knowledge_base():
    global kb_df
    try:
        kb_df = pd.read_csv(KB_FILE)
        if embedding_model:
            print("🧠 Embedding Knowledge Base...")
            # Create a combined text column for embedding
            kb_df['combined_text'] = kb_df['use_case'] + ": " + kb_df['best_practice']
            embeddings = genai.embed_content(model=embedding_model, content=kb_df['combined_text'].tolist(), task_type="retrieval_document")
            kb_df['embedding'] = embeddings['embedding']
            print("   - ✅ Knowledge Base embedded successfully.")
        else:
            print("   - ⚠️  Embedding model not available.")
    except Exception as e:
        print(f"❌ Failed to initialize Knowledge Base: {e}")
        kb_df = None

# --- Helper Functions ---
def update_task(task_id, status=None, message=None, progress=None, result=None):
    if task_id in tasks:
        if status: tasks[task_id]['status'] = status
        if message:
            timestamp = time.strftime('%H:%M:%S')
            tasks[task_id]['logs'].append(f"[{timestamp}] {message}")
        if progress is not None: tasks[task_id]['progress'] = progress
        if result: tasks[task_id]['result'] = result

# --- REWRITTEN: RAG Context Retrieval using Vector Search ---
def retrieve_relevant_context(findings_summary_text, top_k=3):
    if kb_df is None or 'embedding' not in kb_df.columns or not embedding_model:
        return "Knowledge Base not available for context retrieval."
    try:
        findings_embedding = genai.embed_content(model=embedding_model, content=findings_summary_text, task_type="retrieval_query")['embedding']
        
        # Using numpy for a more robust cosine similarity calculation
        kb_embeddings = np.array(kb_df['embedding'].tolist())
        query_embedding = np.array(findings_embedding)
        
        dot_products = np.dot(kb_embeddings, query_embedding)
        kb_norms = np.linalg.norm(kb_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        similarities = dot_products / (kb_norms * query_norm)
        kb_df['similarity'] = similarities
        
        top_k_df = kb_df.nlargest(top_k, 'similarity')
        
        context = [f"- Use Case: {row['use_case']}\n  Best Practice: {row['best_practice']}" for _, row in top_k_df.iterrows()]
        
        return "\n".join(context) if context else "No specifically relevant best practices found."
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return "Error retrieving context from knowledge base."


def generate_recommendations_with_llm(task_id, score_report):
    update_task(task_id, message="[Agent 2] Analyzing score report with RAG engine...")
    if not llm_model:
        update_task(task_id, message="   - ⚠️ LLM not configured. Skipping recommendations.")
        return ["LLM model not available. Recommendations could not be generated."]

    # --- NEW: Helper function to sanitize LLM output ---
    def sanitize_recommendations(data):
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            return [str(value) for value in data.values()]
        if isinstance(data, str):
            return [data]
        return ["Could not parse recommendations from the AI model."]

    findings_summary = [f"{item['name']} (Score: {item['score']:.0f}/{item['max_score']})" for item in score_report['breakdown'] if item['score'] > 0]
    findings_summary_text = ", ".join(findings_summary)
    rag_context = retrieve_relevant_context(findings_summary_text)
    update_task(task_id, message=f"[Agent 2] Retrieved relevant context from knowledge base.")

    prompt = f"""
    You are Agent 2, an expert Personalization Advisor. Agent 1 found: {findings_summary_text} (Overall Score: {score_report['total_score']:.0f}/100).
    **Your Task:** Use the following best practices to inform your response. Your recommendations MUST be based on these practices.
    **Relevant Best Practices:**
    {rag_context}
    Generate a JSON object with a key "recommendations", which is a list of 3-4 actionable recommendations. Justify each recommendation by referencing a best practice.
    """
    try:
        response = llm_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        llm_result = json.loads(cleaned_response)
        update_task(task_id, message="   - ✅ Recommendations generated successfully.")
        
        # Use the sanitizer to guarantee a clean list is returned
        raw_recs = llm_result.get('recommendations', [])
        return sanitize_recommendations(raw_recs)

    except Exception as e:
        update_task(task_id, message=f"   - ❌ LLM Error: {e}")
        return [f"An error occurred while generating recommendations: {e}"]

# (All other functions like create_and_save_presentation, detect_vendors, run_deep_simulation, etc. remain exactly the same)
# ...

def create_and_save_presentation(result_data, filename):
    prs = Presentation()
    score_report = result_data.get('score_report', {})
    # This now safely assumes recommendations is a list of strings
    recommendations = result_data.get('recommendations', [])
    
    # Slide 1: Title
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title, subtitle = slide.shapes.title, slide.placeholders[1]
    url = score_report.get("url", "Website")
    title.text = "Personalization Audit Report"
    subtitle.text = f"An analysis of {url}\nGenerated on {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    
    # Slide 2: Score Summary
    slide = prs.slides.add_slide(prs.slide_layouts[5]) # Blank layout
    title_shape = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(1))
    title_shape.text = "Executive Summary: Score Breakdown"
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.size = Pt(28)
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5.5))
    tf = txBox.text_frame
    tf.text = f"Overall Score: {score_report.get('total_score', 0):.0f}/100"
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.size = Pt(24)
    for item in score_report.get('breakdown', []):
        p = tf.add_paragraph()
        p.text = f"{item['name']}: {item['score']:.0f}/{item['max_score']} - {item['details']}"
        p.level = 1
        
    # Create one slide per recommendation (no type check needed now)
    for i, rec in enumerate(recommendations, 1):
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = f"Recommendation #{i}"
        tf = slide.placeholders[1].text_frame
        tf.clear()
        p = tf.add_paragraph()
        p.text = rec # Assumed to be a string
        p.font.size = Pt(18)
        
    # Save to file
    if not os.path.exists('reports'):
        os.makedirs('reports')
    prs.save(filename)


def detect_vendors(driver, task_id):
    update_task(task_id, message="[Agent 1] Scanning for A/B testing and personalization tools...")
    try: vendors_df = pd.read_csv(VENDORS_FILE)
    except FileNotFoundError:
        update_task(task_id, message=f"   - ⚠️  Warning: '{VENDORS_FILE}' not found. Skipping tool detection.")
        return []
    found_vendors = []
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
                vendor_name = f"{vendor['vendor_name']} ({vendor['category']})"
                if vendor_name not in found_vendors: found_vendors.append(vendor_name)
    if found_vendors: update_task(task_id, message=f"   - ✅ Found tools: {', '.join(found_vendors)}")
    else: update_task(task_id, message="   - ⚪ No specific vendor tools detected.")
    return found_vendors
def check_for_recommendations(driver, task_id):
    try:
        body_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
        for keyword in RECOMMENDATION_KEYWORDS:
            if keyword in body_text:
                update_task(task_id, message=f"   - ✅ Found recommendation keyword: '{keyword}'")
                return True
    except Exception: return False
    return False
def find_product_links(driver, base_url, task_id, limit=3):
    update_task(task_id, message="[Agent 1] Searching for product links...")
    link_patterns = ['/products/', '/p/', '/dp/', '/item/']
    links = []
    WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
    all_links = driver.find_elements(By.TAG_NAME, 'a')
    for a in all_links:
        href = a.get_attribute('href')
        if href and any(pattern in href for pattern in link_patterns):
            if href.startswith('http') and urlparse(href).netloc == urlparse(base_url).netloc:
                if href not in links: links.append(href)
    if not links:
        update_task(task_id, message="   - No common product patterns found. Falling back to any valid link.")
        for a in all_links:
            href = a.get_attribute('href')
            if href and href.startswith('http') and urlparse(href).netloc == urlparse(base_url).netloc and href != base_url:
                if href not in links: links.append(href)
    update_task(task_id, message=f"   - Found {len(links)} potential links.")
    return links[:limit]
def compare_images(path_before, path_after, diff_path, task_id):
    try:
        before_img = Image.open(path_before).convert('RGB')
        width, height = before_img.size
        if width < 7 or height < 7:
            update_task(task_id, message=f"   - ⚠️  Warning: Screenshot is too small ({width}x{height} pixels). Skipping comparison.")
            return 0.0
        after_img = Image.open(path_after).convert('RGB')
        if before_img.size != after_img.size: after_img = after_img.resize(before_img.size)
        diff_img = ImageChops.difference(before_img, after_img)
        diff_img.save(diff_path)
        before_arr, after_arr = np.array(before_img), np.array(after_img)
        similarity_score = ssim(before_arr, after_arr, channel_axis=-1, data_range=before_arr.max() - before_arr.min())
        return (1 - similarity_score) * 100
    except FileNotFoundError: return 0
def run_deep_simulation(url, task_id, num_products=3):
    update_task(task_id, status='running', message="--- Starting Deep Agent Simulation ---", progress=5)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless'); options.add_argument("--window-size=1920,1080"); options.add_argument('--no-sandbox'); options.add_argument('--disable-dev-shm-usage'); options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)
    if not os.path.exists('screenshots'): os.makedirs('screenshots')
    score = { 'homepage_change': 0, 'tools_detected': 0, 'homepage_recs': 0, 'product_recs': 0, 'cart_recs': 0 }
    found_vendors_list = []
    difference = 0.0
    try:
        update_task(task_id, message="[Agent 1] Initializing...", progress=10)
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        before_screenshot_path = f'screenshots/{task_id}_before.png'
        driver.save_screenshot(before_screenshot_path)
        update_task(task_id, message="[Agent 1] Took 'before' screenshot.")
        found_vendors_list = detect_vendors(driver, task_id)
        if found_vendors_list: score['tools_detected'] = 25
        update_task(task_id, progress=25)
        product_links = find_product_links(driver, url, task_id, limit=num_products)
        if not product_links: raise Exception("Could not find any product links to visit.")
        for i, link in enumerate(product_links, 1):
            update_task(task_id, message=f"[Agent 1] Visiting product {i}/{len(product_links)}...")
            driver.get(link)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            if score['product_recs'] == 0 and check_for_recommendations(driver, task_id): score['product_recs'] = 15
        update_task(task_id, progress=50)
        cart_paths = ['/cart', '/basket', '/checkout/cart']
        cart_found = False
        for path in cart_paths:
            cart_url = url.rstrip('/') + path
            try:
                driver.get(cart_url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                if "cart" in driver.title.lower() or "basket" in driver.title.lower():
                    cart_found = True
                    if check_for_recommendations(driver, task_id): score['cart_recs'] = 15
                    break
            except Exception: continue
        update_task(task_id, progress=65)
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        after_screenshot_path = f'screenshots/{task_id}_after.png'
        driver.save_screenshot(after_screenshot_path)
        update_task(task_id, message="[Agent 1] Took 'after' screenshot.")
        if check_for_recommendations(driver, task_id): score['homepage_recs'] = 15
        update_task(task_id, progress=80)
        diff_path = f'screenshots/{task_id}_diff.png'
        difference = compare_images(before_screenshot_path, after_screenshot_path, diff_path, task_id)
        if difference >= 5.0: score['homepage_change'] = 30.0
        else: score['homepage_change'] = (difference / 5.0) * 30.0
        update_task(task_id, message="[Agent 1] Quantitative analysis complete.")
        total_score = sum(score.values())
        score_report = {
            "url": url, "total_score": total_score,
            "breakdown": [
                {"name": "Dynamic Homepage Change", "score": score['homepage_change'], "max_score": 30, "details": f"Based on {difference:.2f}% visual change"},
                {"name": "A/B & Personalization Tools", "score": score['tools_detected'], "max_score": 25, "details": f"Found: {', '.join(found_vendors_list)}" if found_vendors_list else "None"},
                {"name": "Homepage Recommendations", "score": score['homepage_recs'], "max_score": 15, "details": ""},
                {"name": "Product Page Recommendations", "score": score['product_recs'], "max_score": 15, "details": ""},
                {"name": "Cart Page Recommendations", "score": score['cart_recs'], "max_score": 15, "details": ""},
            ]
        }
        update_task(task_id, progress=90)
        recommendations = generate_recommendations_with_llm(task_id, score_report)
        final_result = { "score_report": score_report, "recommendations": recommendations }
        print(recommendations)
        update_task(task_id, message="[System] Generating PowerPoint report...", progress=98)
        presentation_filename = f"reports/Report_{task_id}.pptx"
        create_and_save_presentation(final_result, presentation_filename)
        update_task(task_id, message=f"[System] Report saved as {presentation_filename}")
        update_task(task_id, status='completed', message="--- All Agents Finished ---", progress=100, result=final_result)
    except Exception as e:
        update_task(task_id, status='failed', message=f"❌ An error occurred: {e}", progress=100)
    finally:
        driver.quit()
# --- Flask App and API Endpoints ---
app = Flask(__name__)
CORS(app)
@app.route('/audit', methods=['POST'])
def start_audit():
    url = request.json.get('url')
    if not url: return jsonify({"error": "URL is required"}), 400
    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'pending', 'progress': 0, 'logs': [], 'result': None}
    thread = threading.Thread(target=run_deep_simulation, args=(url, task_id))
    thread.daemon = True
    thread.start()
    return jsonify({"task_id": task_id})
@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task: return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

if __name__ == '__main__':
    # Embed the knowledge base once on startup
    initialize_knowledge_base()
    app.run(debug=True, port=5001)