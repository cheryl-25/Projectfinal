'''
SCRAPER.PY - Web Scraper for DeKUT Website
'''

import requests
from bs4 import BeautifulSoup
import json
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DeKUTScraper:
    def __init__(self):
        """
        Initialize the scraper with:
        - base_url: DeKUT website
        - session: Maintain connection for multiple requests
        - headers: Pretend to be a browser (some websites block bots)
        """
        self.base_url = "https://dkut.ac.ke"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_page(self, page_url, keywords):
        """
        Scrape a single page and extract relevant information
        
        Args:
            page_url: Specific page to scrape (e.g., "/admissions")
            keywords: List of words to look for in content
        
        Returns:
            List of FAQs found on that page
        """
        print(f"📡 Scraping: {self.base_url}{page_url}")
        
        try:
            # 1. Fetch the webpage
            response = self.session.get(self.base_url + page_url, timeout=10,verify=False)
            
            # 2. Check if successful
            if response.status_code != 200:
                print(f"❌ Failed to fetch page: {response.status_code}")
                return []
            
            # 3. Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            faqs = []
            
            # 4. Look for all headings (h1-h6)
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            for heading in headings:
                heading_text = heading.text.strip().lower()
                
                # 5. Check if heading contains any keywords we're interested in
                if any(keyword in heading_text for keyword in keywords):
                    
                    # 6. Get content after the heading
                    content = ""
                    next_element = heading.find_next_sibling()
                    
                    # 7. Collect all text until next heading
                    while next_element and next_element.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        if next_element.name == 'p':  # Only get paragraph text
                            content += next_element.get_text() + " "
                        next_element = next_element.find_next_sibling()
                    
                    # 8. Only add if we have meaningful content
                    if content.strip() and len(content) > 20:
                        faq = {
                            "question": heading.text.strip(),
                            "answer": content.strip()[:300],  # Limit to 300 chars
                            "source": page_url
                        }
                        faqs.append(faq)
                        print(f"   ✅ Found: {heading.text[:50]}...")
            
            print(f"   📊 Found {len(faqs)} FAQs")
            return faqs
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def scrape_all(self):
        """
        Scrape multiple important pages from DeKUT website
        
        Returns:
            Combined list of all FAQs from all pages
        """
        all_faqs = []
        
        # Define which pages to scrape and what keywords to look for
        pages_to_scrape = [
            {
                "url": "/index.php/academics/admission/application-forms",
                "keywords": ["admission", "apply", "requirement", "deadline", "application"]
            },
            {
                "url":"/index.php/students/dekut-financial-aid-office-defao",
                "keywords":["aid","support","needs","financial aid"]
            },
            {
                "url": "/index.php/about-dekut/s5-accordion-menu/location-and-contacts",
                "keywords": ["contact", "phone", "email", "address", "location"]
            }
        ]
        
        print("🚀 Starting DeKUT Website Scraping...")
        print("=" * 50)
        
        # 9. Scrape each page one by one
        for page_info in pages_to_scrape:
            faqs = self.scrape_page(page_info["url"], page_info["keywords"])
            all_faqs.extend(faqs)
            time.sleep(1)  # Wait 1 second between requests (be polite to server)
        
        print("=" * 50)
        print(f"🎉 Total FAQs scraped: {len(all_faqs)}")
        
        # 10. Save scraped data to file
        self.save_to_file(all_faqs)
        
        return all_faqs
    
    def save_to_file(self, faqs):
        """Save scraped FAQs to JSON file"""
        if faqs:
            with open('data/scraped_faqs.json', 'w', encoding='utf-8') as f:
                json.dump(faqs, f, indent=2, ensure_ascii=False)
            print("💾 Saved to: data/scraped_faqs.json")
    
    def update_intents(self):
        """
        Merge scraped FAQs with existing intents.json
        This creates a hybrid dataset (manual + scraped)
        """
        # 11. Load existing intents
        with open('intents.json', 'r', encoding='utf-8') as f:
            intents = json.load(f)
        
        # 12. Scrape new data
        scraped_faqs = self.scrape_all()
        
        if not scraped_faqs:
            print("⚠️ No data scraped. Using existing intents only.")
            return
        
        # 13. Create a new intent for scraped data
        scraped_intent = {
            "tag": "scraped_faqs",
            "patterns": [],
            "responses": []
        }
        
        # 14. Convert each scraped FAQ to training format
        for faq in scraped_faqs:
            question = faq['question']
            answer = faq['answer']
            
            # Create 3 pattern variations for each question
            patterns = [
                question,
                question.lower(),
                f"what is {question.lower().replace('?', '')}",
                f"tell me about {question.lower().replace('?', '')}",
                question.replace("?", " please")
            ]
            
            # Add patterns and response
            scraped_intent['patterns'].extend(patterns)
            scraped_intent['responses'].append(answer)
        
        # 15. Remove duplicates
        scraped_intent['patterns'] = list(set(scraped_intent['patterns']))
        
        # 16. Add to intents list
        intents['intents'].append(scraped_intent)
        
        # 17. Save updated intents
        with open('intents.json', 'w', encoding='utf-8') as f:
            json.dump(intents, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Added {len(scraped_faqs)} scraped FAQs to intents.json")
        print(f"📈 Total intents: {len(intents['intents'])}")

# 18. This runs when you execute: python scraper.py
if __name__ == "__main__":
    print("🤖 DeKUT Website Scraper")
    print("-" * 30)
    
    scraper = DeKUTScraper()
    
    choice = input("Do you want to scrape now? (y/n): ").lower()
    if choice == 'y':
        scraper.update_intents()
    else:
        print("Scraping skipped. Using existing intents.json")