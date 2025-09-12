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
from PIL import Image, ImageChops
import google.generativeai as genai

# --- In-memory storage for running tasks ---
tasks = {}

# --- Configuration for Scoring & RAG ---
VENDORS_FILE = 'vendors.csv'
KB_FILE = 'personalization_kb.csv'
RECOMMENDATION_KEYWORDS = [
    'recommended for you', 'inspired by your browsing', 'recently viewed',
    'customers also bought', 'frequently bought together', 'similar products',
    'you might also like', 'because you viewed', 'complete your purchase'
]

# --- RAG/LLM Setup ---
try:
    kb_df = pd.read_csv(KB_FILE)
    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"⚠️ RAG/LLM Warning: {e}")
    llm_model = None

# --- Helper Functions ---
def update_task(task_id, status=None, message=None, progress=None, result=None):
    if task_id in tasks:
        if status: tasks[task_id]['status'] = status
        if message:
            timestamp = time.strftime('%H:%M:%S')
            tasks[task_id]['logs'].append(f"[{timestamp}] {message}")
        if progress is not None: tasks[task_id]['progress'] = progress
        if result: tasks[task_id]['result'] = result

# --- THIS IS THE CORRECTED FUNCTION ---
def get_rag_context(score_report):
    """Retrieves relevant best practices from the knowledge base for the LLM."""
    context = []
    # Create a simple map from the score report's breakdown for easy lookups
    findings_map = {item['name']: item for item in score_report.get('breakdown', [])}

    # Check for findings using the new structure
    if findings_map.get('A/B & Personalization Tools', {}).get('score', 0) > 0:
        context.append(kb_df[kb_df['use_case'] == 'Product Recommendations'].iloc[0]['best_practice'])
    if findings_map.get('Homepage Recommendations', {}).get('score', 0) > 0:
        context.append(kb_df[kb_df['use_case'] == 'Personalized Hero Banners'].iloc[0]['best_practice'])
    
    return "\n".join(context) if context else "General personalization best practices."

def generate_recommendations_with_llm(task_id, score_report):
    """Calls the LLM to generate qualitative recommendations based on quantitative findings."""
    update_task(task_id, message="[Agent 2] Analyzing score report with RAG engine...")
    if not llm_model:
        update_task(task_id, message="   - ⚠️ LLM not configured. Skipping recommendations.")
        return ["LLM model not available. Recommendations could not be generated."]

    findings_summary = []
    for item in score_report['breakdown']:
        if item['score'] > 0:
            findings_summary.append(f"- {item['name']}: Scored {item['score']:.0f}/{item['max_score']}. {item['details']}")
    
    rag_context = get_rag_context(score_report)

    prompt = f"""
    You are Agent 2, a world-class Personalization Advisor.
    Agent 1 has conducted a technical analysis of a website and produced the following score report:

    **Quantitative Findings from Agent 1:**
    - Overall Score: {score_report['total_score']:.0f}/100
    - Breakdown:
    {json.dumps(findings_summary, indent=2)}

    **Relevant Best Practices from RAG Knowledge Base:**
    {rag_context}

    **Your Task:**
    Based on the quantitative findings and best practices, generate a JSON object containing a key "recommendations". This should be a list of 3 to 4 concise, actionable, and expert-level recommendations to help the user improve their personalization strategy.
    """
    try:
        response = llm_model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        llm_result = json.loads(cleaned_response)
        update_task(task_id, message="   - ✅ Recommendations generated successfully.")
        return llm_result.get('recommendations', [])
    except Exception as e:
        update_task(task_id, message=f"   - ❌ LLM Error: {e}")
        return [f"An error occurred while generating recommendations: {e}"]

# --- Functions from deep_agent.py (adapted for background tasks) ---
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
        # --- AGENT 1 EXECUTION ---
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

        # --- AGENT 2 EXECUTION ---
        total_score = sum(score.values())
        score_report = {
            "total_score": total_score,
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
        
        final_result = {
            "score_report": score_report,
            "recommendations": recommendations
        }
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
    app.run(debug=True, port=5001)