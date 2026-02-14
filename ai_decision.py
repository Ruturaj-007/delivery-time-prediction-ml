import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not configured in .env file")

# Delivery performance summary
summary = {
    "context": "Food delivery service performance evaluation",
    "signals": {
        "total_deliveries": 1000,
        "avg_delivery_time": 27.3,
        "delayed_deliveries_percentage": 8.2,
        "top_performer_rating": 4.8,
        "avg_partner_rating": 4.3,
        "longest_distance_avg": 12.5
    }
}

def analyze_decision():
    """Analyze delivery data using Gemini AI and return decision"""
    
    prompt = f"""
You are a senior AI decision assistant for a food delivery company.

Context:
{summary['context']}

Computed Signals:
{json.dumps(summary['signals'], indent=2)}

Your task:
1. Classify overall delivery performance as GOOD / WARNING / CRITICAL
2. Explain the decision in simple business language
3. Suggest ONE immediate action to improve delivery times
4. Suggest ONE long-term strategy to optimize the delivery network

Return STRICT JSON ONLY:
{{
  "status": "",
  "reason": "",
  "immediate_action": "",
  "long_term_recommendation": ""
}}
"""

    # ✅ Using EXACT model from your TypeScript code: gemini-2.5-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        
        ai_response = response.json()
        
        response_text = ai_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not response_text:
            raise ValueError("Empty response from Gemini")
        
        # Clean up response
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        decision = json.loads(response_text)
        
        print("✅ AI Decision Output:")
        print(json.dumps(decision, indent=2))
        
        return decision
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        print(f"🔑 Check your API key: {GEMINI_API_KEY[:10]}...")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Raw response: {response_text}")
        raise

if __name__ == "__main__":
    try:
        analyze_decision()
    except Exception as e:
        print(f"Error: {e}")