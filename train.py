

import json
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SimpleIntentClassifier:
    def __init__(self, intents_file='intents.json'):
        self.intents_file = intents_file
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.label_encoder = LabelEncoder()
        self.classifier = None
    
    def load_intents(self):
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def prepare_data(self):
        """Prepare training data from intents.json"""
        data = self.load_intents()
        
        patterns = []
        tags = []
        
        for intent in data['intents']:
            for pattern in intent['patterns']:
                patterns.append(pattern.lower())
                tags.append(intent['tag'])
        
        return patterns, tags
    
    def train(self):
        """Train the classifier"""
        print("🤖 Training Simple Intent Classifier...")
        print("=" * 50)
        
        # 1. Load data
        patterns, tags = self.prepare_data()
        
        print(f"📊 Training samples: {len(patterns)}")
        print(f"📊 Unique intents: {len(set(tags))}")
        
        # 2. Convert text to numbers (TF-IDF)
        X = self.vectorizer.fit_transform(patterns)
        
        # 3. Convert labels to numbers
        y = self.label_encoder.fit_transform(tags)
        
        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 5. Train classifier (using Logistic Regression - fast and accurate)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # 6. Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Training Accuracy: {accuracy:.2%}")
        
        # 7. Save everything
        self.save_model()
        
        return accuracy
    
    def save_model(self):
        """Save trained model"""
        import os
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Save vectorizer
        with open('models/tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save label encoder
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save classifier
        joblib.dump(self.classifier, 'models/intent_classifier.joblib')
        
        print("💾 Models saved to 'models/' folder:")
        print("  ✅ tfidf_vectorizer.pkl")
        print("  ✅ label_encoder.pkl")
        print("  ✅ intent_classifier.joblib")
    
    def predict(self, text):
        """Predict intent for a single text"""
        # Transform text
        X = self.vectorizer.transform([text.lower()])
        
        # Predict
        prediction = self.classifier.predict(X)
        probability = np.max(self.classifier.predict_proba(X))
        
        # Decode label
        intent = self.label_encoder.inverse_transform(prediction)[0]
        
        return intent, probability

if __name__ == "__main__":
    try:
        # 1. Initialize the classifier
        # Make sure 'intents.json' is in the same folder!
        classifier = SimpleIntentClassifier(intents_file='intents.json')
        
        # 2. Run the training
        classifier.train()
        
        # 3. Quick test (Optional)
        print("\n🔎 Quick Test:")
        test_text = "Hello there"
        intent, prob = classifier.predict(test_text)
        print(f"   Input: '{test_text}' -> Detected: {intent} ({prob:.2%})")
        
    except FileNotFoundError:
        print("❌ Error: 'intents.json' file not found. Please make sure it exists in this folder.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")