"""
SIMPLE APP.PY - Uses scikit-learn model instead of TensorFlow
"""

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pickle
import joblib
import random

app = Flask(__name__)

class SimpleDekutChatbot:
    def __init__(self):
        print("🤖 Loading Simple DeKUT Chatbot...")
        
        try:
            # Load intents
            with open('intents.json', 'r') as f:
                self.intents_data = json.load(f)
            
            # Load models
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('models/label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            self.classifier = joblib.load('models/intent_classifier.joblib')
            
            # Create response mapping
            self.tag_responses = {}
            for intent in self.intents_data['intents']:
                self.tag_responses[intent['tag']] = intent['responses']
            
            print(f"✅ Loaded {len(self.tag_responses)} intent categories")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n💡 Run this first:")
            print("1. python scraper.py")
            print("2. python train.py")
            raise
    
    def get_response(self, user_input):
        """Get response for user input"""
        if not user_input.strip():
            return "Please type a question."
        
        try:
            # Transform input
            X = self.vectorizer.transform([user_input.lower()])
            
            # Predict
            prediction = self.classifier.predict(X)

            
            probability = np.max(self.classifier.predict_proba(X))
            print(f"prediction {probability} ")
            # Decode intent
            intent = self.label_encoder.inverse_transform(prediction)[0]
            print(f"prediction {intent} ")
            print(f"🎯 User: '{user_input}'")
            print(f"🎯 Intent: {intent} (confidence: {probability:.2%})")
            
            # Check confidence threshold
            if probability < 0.25 and intent in self.tag_responses:
                return random.choice(self.tag_responses[intent])
            else:
                fallbacks = [
                    "I'm not sure about that. Please visit https://dkut.ac.ke",
                    "Could you rephrase your question?",
                    "I'm still learning about that topic.",
                    "Please contact admissions@dkut.ac.ke for more information."
                ]
                return random.choice(fallbacks)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return "I'm having trouble understanding. Please try again."

# Initialize chatbot
chatbot = SimpleDekutChatbot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=["POST"])
def chat():
    user_message = request.json.get('message', '').strip()
    
    if not user_message:
        return jsonify({'response': 'Please enter a message.'})
    
    response = chatbot.get_response(user_message)
    return jsonify({'response': response, 'success': True})

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 DeKUT Chatbot Web Application")
    print("=" * 50)
    print("\n🌐 Open: http://localhost:5000")
    print("🛑 Press CTRL+C to stop\n")
    
    app.run(debug=True, port=5000)