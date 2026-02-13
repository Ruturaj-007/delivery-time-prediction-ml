import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not configured in .env file")

# Summary data structure
summary = {
    "context": "Student academic performance evaluation dataset",
    "signals": {
        "total_students": 1000,
        "overall_avg_score": 67.71,
        "risk_percentage": 12.4,
        "test_preparation_impact": {
            "completed": 72.82,
            "none": 65.14
        },
        "weakest_subject": "math score"
    }
}

def analyze_decision():
    """Analyze academic data using Gemini AI and return decision"""
    
    prompt = f"""
You are a senior AI decision assistant used by school leadership.

Context:
{summary['context']}

Computed Signals:
{json.dumps(summary['signals'], indent=2)}

Your task:
1. Classify overall academic status as GOOD / WARNING / CRITICAL
2. Explain the decision in simple leadership language
3. Suggest ONE immediate action
4. Suggest ONE long-term improvement strategy

Return STRICT JSON ONLY:
{{
  "status": "",
  "reason": "",
  "immediate_action": "",
  "long_term_recommendation": ""
}}
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    
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
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        ai_response = response.json()
        
        response_text = ai_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not response_text:
            raise ValueError("Empty response from Gemini")
        
        # Clean up response
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        decision = json.loads(response_text)
        
        print("AI Decision Output:")
        print(json.dumps(decision, indent=2))
        
        return decision
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON Parse Error: {e}")
        print(f"Raw response: {response_text}")
        raise

if __name__ == "__main__":
    try:
        analyze_decision()
    except Exception as e:
        print(f"Error: {e}")