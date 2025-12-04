import json
import random
import requests
import urllib3
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
app = Flask(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🛠️ SETTINGS: Keep this False to test Scraper & Intents first!
ENABLE_AI = False 
GENAI_API_KEY = "YOUR_API_KEY_HERE" # (We will use this in the next step)

# --- 1. LOAD INTENTS ---
try:
    with open('intents.json', 'r') as file:
        intents_data = json.load(file)
    print(f"✅ Intents Loaded: {len(intents_data['intents'])} categories.")
except Exception as e:
    print(f"❌ Error loading intents.json: {e}")
    intents_data = {"intents": []}

# --- 2. THE SCRAPER (Knowledge Base) ---
# We are scraping these 2 reliable pages first
URLS_TO_SCRAPE = [
    "https://www.dkut.ac.ke/", 
    "https://www.dkut.ac.ke/index.php/admission/admission-requirements"
]

def get_school_data():
    print("🕷️  Scraping DeKUT website... (Please wait)")
    data = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'} 
    
    for url in URLS_TO_SCRAPE:
        try:
            page = requests.get(url, headers=headers, verify=False, timeout=15)
            soup = BeautifulSoup(page.content, "html.parser")
            
            # Remove junk (menus, footers, scripts)
            for script in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                script.decompose()
            
            # Extract text and clean it
            text = soup.get_text(separator=' ')
            # Only keep "meaty" sentences (> 60 chars)
            chunks = [t.strip() for t in text.split('\n') if len(t.strip()) > 60]
            data.extend(chunks)
        except Exception as e:
            print(f"⚠️  Skipping {url}: {e}")
            
    print(f"✅ Knowledge Base Loaded: {len(data)} text chunks.")
    return data

# Load scraper data once when app starts
KNOWLEDGE_BASE = get_school_data()

# --- 3. LOGIC BRAIN ---

def get_intent_match(user_input):
    """Checks your handwritten intents.json first."""
    user_input = user_input.lower()
    for intent in intents_data['intents']:
        for pattern in intent['text']:
            if pattern.lower() in user_input:
                return random.choice(intent['responses'])
    return None

def find_best_context(user_query):
    """Searches the scraped website data."""
    if not KNOWLEDGE_BASE:
        return None, 0.0
    
    try:
        documents = [user_query] + KNOWLEDGE_BASE
        tfidf = TfidfVectorizer().fit_transform(documents)
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

    # STEP 1: Check Intents (Highest Priority)
    intent_reply = get_intent_match(user_text)
    if intent_reply:
        print("   ✅ Matched Intent")
        return jsonify({"response": intent_reply})
    
    # STEP 2: Check Scraper (The Website)
    context, score = find_best_context(user_text)
    print(f"   🔍 Scraper Score: {score}")

    # Strict Threshold: Only answer if we are sure (> 0.15)
    if score > 0.15:
        if ENABLE_AI:
            # TODO: We will enable this in Phase 2
            return jsonify({"response": "AI Mode is not active yet."})
        else:
            # VALIDATION MODE: Return the exact text found on the site
            return jsonify({"response": f"📚 <b>Found on Website:</b><br>{context}"})
    
    # STEP 3: Fallback (Prevents Hallucination)
    print("   ⛔ Low Score - Blocking Hallucination")
    return jsonify({"response": "I'm sorry, I don't have information on that specific topic yet. Please try asking about admissions, courses, or fees."})

if __name__ == "__main__":
    app.run(debug=True, port=8080)