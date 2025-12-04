import json
import random
import requests
import urllib3
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import os

# --- CONFIGURATION ---
app = Flask(__name__)
# Disable SSL warnings for scraping
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🛠️ SETTINGS
ENABLE_AI = False 
GENAI_API_KEY = "YOUR_API_KEY_HERE" # (Phase 2 feature)

# --- 1. LOAD INTENTS ---
try:
    # Use os.path to find the file reliably
    work_dir = os.path.dirname(os.path.abspath(__file__))
    intents_path = os.path.join(work_dir, 'intents.json')
    
    with open(intents_path, 'r') as file:
        intents_data = json.load(file)
    print(f"✅ Intents Loaded: {len(intents_data['intents'])} categories.")
except Exception as e:
    print(f"❌ Error loading intents.json: {e}")
    intents_data = {"intents": []}

# --- 2. THE UPGRADED SCRAPER (Knowledge Base) ---
# Expanded list of URLs to cover Fees, Courses, and Contacts
URLS_TO_SCRAPE = [
    "https://www.dkut.ac.ke/", 
    "https://www.dkut.ac.ke/index.php/about-dekut/s5-accordion-menu/our-profile",
    "https://www.dkut.ac.ke/index.php/academics/undergraduate-programmes",
    "https://www.dkut.ac.ke/index.php/admission/admission-requirements",
    "https://www.dkut.ac.ke/index.php/contact-us"
]

def get_school_data():
    print("🕷️  Scraping DeKUT website... (This may take a few seconds)")
    data = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} 
    
    for url in URLS_TO_SCRAPE:
        try:
            page = requests.get(url, headers=headers, verify=False, timeout=15)
            soup = BeautifulSoup(page.content, "html.parser")
            
            # 1. Kill junk elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
                script.decompose()
            
            # 2. TARGETED SCRAPING: Get text from specific content tags
            # We look for Paragraphs <p>, List Items <li>, Headers <h>, and Table Cells <td>
            content_tags = soup.find_all(['p', 'li', 'td', 'h1', 'h2', 'h3', 'div'])
            
            for tag in content_tags:
                text = tag.get_text(strip=True)
                # Keep sentences that have at least 5 words to avoid menus/buttons
                if len(text.split()) > 5:
                    data.append(text)
            
            print(f"   -> Successfully read {url}")
            
        except Exception as e:
            print(f"⚠️  Skipping {url}: {e}")
            
    # Remove duplicates to save memory
    unique_data = list(set(data))
    print(f"✅ Knowledge Base Loaded: {len(unique_data)} text chunks.")
    return unique_data

# Load scraper data once when app starts
KNOWLEDGE_BASE = get_school_data()

# --- 3. LOGIC BRAIN ---

def get_intent_match(user_input):
    """Checks your handwritten intents.json first."""
    user_input = user_input.lower()
    for intent in intents_data['intents']:
        for pattern in intent['text']:
            # Exact match or close phrase match
            if pattern.lower() in user_input:
                return random.choice(intent['responses'])
    return None

def find_best_context(user_query):
    """Searches the scraped website data."""
    if not KNOWLEDGE_BASE:
        return None, 0.0
    
    try:
        # Add user query to the data to train the vectorizer on the fly
        documents = [user_query] + KNOWLEDGE_BASE
        tfidf = TfidfVectorizer().fit_transform(documents)
        
        # Compare user query (index 0) against all other documents (index 1 to end)
        cosine_similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        
        best_index = cosine_similarities.argmax()
        score = cosine_similarities[best_index]
        
        return KNOWLEDGE_BASE[best_index], score
    except Exception as e:
        print(f"Vector Error: {e}")
        return None, 0.0

# --- 4. FLASK ROUTES ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_text = request.form["msg"]
    print(f"\n📩 User asked: {user_text}")

    # STEP 1: Check Intents (Highest Priority - Fast & Accurate for greetings)
    intent_reply = get_intent_match(user_text)
    if intent_reply:
        print("   ✅ Matched Intent")
        return jsonify({"response": intent_reply})
    
    # STEP 2: Check Scraper (The Website)
    context, score = find_best_context(user_text)
    
    # Debug print to see what the bot found
    if context:
        print(f"   🔍 Best Match: {context[:50]}... | Score: {score:.2f}")

    # STEP 3: Strict Threshold to stop Hallucinations
    # We increased this from 0.15 to 0.35 to ensure quality matches
    if score > 0.35:
        if ENABLE_AI:
            # Phase 2 feature
            return jsonify({"response": "AI Mode is not active yet."})
        else:
            # VALIDATION MODE: Return the exact text found on the site
            return jsonify({"response": f"{context}<br><br><small><i>Source: DeKUT Website</i></small>"})
    
    # STEP 4: Fallback (Prevents answering nonsense)
    print("   ⛔ Low Score - Blocking Hallucination")
    return jsonify({
        "response": "I couldn't find specific information on that in my database. <br>Please check the <a href='https://www.dkut.ac.ke' target='_blank'>Official DeKUT Website</a> or ask about <b>Admissions</b>, <b>Courses</b>, or <b>Fees</b>."
    })

if __name__ == "__main__":
    app.run(debug=True, port=8080)