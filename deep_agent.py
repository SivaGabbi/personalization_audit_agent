import argparse
import time
import os
import pandas as pd
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image, ImageChops

# --- Configuration for Scoring ---
VENDORS_FILE = 'vendors.csv'
RECOMMENDATION_KEYWORDS = [
    'recommended for you', 'inspired by your browsing', 'recently viewed',
    'customers also bought', 'frequently bought together', 'similar products',
    'you might also like', 'because you viewed'
]

# --- Helper Functions for Scoring ---

def detect_vendors(driver):
    """Scans for third-party personalization/testing vendor scripts."""
    print("🔎 Scanning for A/B testing and personalization tools...")
    try:
        vendors_df = pd.read_csv(VENDORS_FILE)
    except FileNotFoundError:
        print(f"   - ⚠️  Warning: '{VENDORS_FILE}' not found. Skipping tool detection.")
        return []

    found_vendors = []
    scripts = driver.execute_script("return Array.from(document.scripts).map(s => s.src);")
    script_sources_text = " ".join(filter(None, scripts))

    for _, vendor in vendors_df.iterrows():
        for signature in vendor['signatures'].split(','):
            is_present = False
            if signature in script_sources_text:
                is_present = True
            else:
                try:
                    if driver.execute_script(f"return typeof {signature} !== 'undefined'"):
                        is_present = True
                except:
                    pass
            
            if is_present:
                vendor_name = f"{vendor['vendor_name']} ({vendor['category']})"
                if vendor_name not in found_vendors:
                    found_vendors.append(vendor_name)
    
    if found_vendors:
        print(f"   - ✅ Found tools: {', '.join(found_vendors)}")
    else:
        print("   - ⚪ No specific vendor tools detected.")
        
    return found_vendors

def check_for_recommendations(driver):
    """Checks the visible text on a page for recommendation-related keywords."""
    try:
        body_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
        for keyword in RECOMMENDATION_KEYWORDS:
            if keyword in body_text:
                print(f"   - ✅ Found recommendation keyword: '{keyword}'")
                return True
    except Exception:
        return False
    return False

# --- Core Simulation and Comparison Logic ---

def find_product_links(driver, base_url, limit=5):
    """Tries to find links that appear to be products."""
    print("🕵️  Searching for product links...")
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
        print("   - No common product patterns found. Falling back to any valid link.")
        for a in all_links:
            href = a.get_attribute('href')
            if href and href.startswith('http') and urlparse(href).netloc == urlparse(base_url).netloc and href != base_url:
                if href not in links: links.append(href)
    print(f"   - Found {len(links)} potential links.")
    return links[:limit]

def compare_images(path_before, path_after, diff_path):
    """Compares two images and returns a difference score."""
    try:
        before_img = Image.open(path_before).convert('RGB')
        width, height = before_img.size
        if width < 7 or height < 7:
            print("   - ⚠️  Warning: Screenshot is too small to analyze. Skipping comparison.")
            return 0.0
        after_img = Image.open(path_after).convert('RGB')
        if before_img.size != after_img.size: after_img = after_img.resize(before_img.size)
        diff_img = ImageChops.difference(before_img, after_img)
        diff_img.save(diff_path)
        before_arr, after_arr = np.array(before_img), np.array(after_img)
        similarity_score = ssim(before_arr, after_arr, channel_axis=-1, data_range=before_arr.max() - before_arr.min())
        return (1 - similarity_score) * 100
    except FileNotFoundError: return 0

# --- Main Simulation Function (Updated with Scoring) ---

def run_deep_simulation(url, num_products=3):
    """
    Simulates a user journey and calculates a detailed personalization score.
    """
    print("\n--- Starting Deep Agent Simulation & Scoring ---")
    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument("--window-size=1920,1080")
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(30)

    if not os.path.exists('screenshots'): os.makedirs('screenshots')

    score = { 'homepage_change': 0, 'tools_detected': 0, 'homepage_recs': 0, 'product_recs': 0 }
    found_vendors_list = []
    
    try:
        print(f"\n[Phase 1: Initial Homepage Analysis]")
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        before_screenshot_path = 'screenshots/homepage_before.png'
        driver.save_screenshot(before_screenshot_path)
        print(f"📸 Took 'before' screenshot.")
        
        found_vendors_list = detect_vendors(driver)
        if found_vendors_list: score['tools_detected'] = 30

        print(f"\n[Phase 2: Product Browsing Simulation]")
        product_links = find_product_links(driver, url, limit=num_products)
        if not product_links:
            print("❌ Could not find any product links to visit. Aborting simulation.")
            return

        for i, link in enumerate(product_links, 1):
            print(f"➡️  Visiting product {i}/{len(product_links)}...")
            driver.get(link)
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            if score['product_recs'] == 0 and check_for_recommendations(driver):
                score['product_recs'] = 15

        print(f"\n[Phase 3: Final Homepage Analysis]")
        driver.get(url)
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        after_screenshot_path = 'screenshots/homepage_after.png'
        driver.save_screenshot(after_screenshot_path)
        print(f"📸 Took 'after' screenshot.")

        if check_for_recommendations(driver): score['homepage_recs'] = 15

        print("\n🔎 Comparing screenshots for dynamic changes...")
        diff_path = 'screenshots/homepage_diff.png'
        difference = compare_images(before_screenshot_path, after_screenshot_path, diff_path)
        
        # --- THIS IS THE MODIFIED SCORING LOGIC ---
        if difference >= 10.0:
            score['homepage_change'] = 40.0
        else:
            score['homepage_change'] = (difference / 10.0) * 40.0

    except Exception as e:
        print(f"\n❌ An error occurred during the simulation: {e}")
    finally:
        driver.quit()

    # --- Final Report ---
    total_score = sum(score.values())
    print("\n" + "="*40)
    print("   Personalization Score Report")
    print("="*40)
    print(f"📊 Total Score: {total_score:.0f} / 100")
    print("\n--- Score Breakdown ---")
    
    change_status = "✅" if score['homepage_change'] > 0 else "⚪"
    print(f"{change_status} Dynamic Homepage Change: {score['homepage_change']:.0f}/40 pts (based on {difference:.2f}% visual change)")

    tools_status = "✅" if score['tools_detected'] > 0 else "⚪"
    tools_details = f"(Found: {', '.join(found_vendors_list)})" if found_vendors_list else ""
    print(f"{tools_status} A/B & Personalization Tools: {score['tools_detected']}/30 pts {tools_details}")
    
    recs_home_status = "✅" if score['homepage_recs'] > 0 else "⚪"
    print(f"{recs_home_status} Homepage Recommendations: {score['homepage_recs']}/15 pts")

    recs_prod_status = "✅" if score['product_recs'] > 0 else "⚪"
    print(f"{recs_prod_status} Product Page Recommendations: {score['product_recs']}/15 pts")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a deep agent simulation to calculate a personalization score.")
    parser.add_argument("url", type=str, help="The base URL of the website to analyze.")
    parser.add_argument("-p", "--products", type=int, default=3, help="The number of product pages to visit (default: 3).")
    args = parser.parse_args()
    
    run_deep_simulation(args.url, args.products)