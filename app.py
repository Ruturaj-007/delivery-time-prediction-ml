import gradio as gr
import pandas as pd
import pickle
import os
import json
import requests
import resend
from dotenv import load_dotenv

load_dotenv()

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "delivery_time_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model_features.pkl"), "rb") as f:
    FEATURES = pickle.load(f)

# ---------------- API KEYS ----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
resend.api_key = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "onboarding@resend.dev")

# ---------------- ENCODING MAPS ----------------
ORDER_MAP = {
    "Buffet": 0,
    "Drinks": 1,
    "Meal": 2,
    "Snack": 3
}

VEHICLE_MAP = {
    "Bicycle": 0,
    "Electric Scooter": 1,
    "Motorcycle": 2,
    "Scooter": 3
}

# ---------------- AI DECISION FUNCTION ----------------
def get_ai_decision():
    """Get AI decision from Gemini"""
    
    if not GEMINI_API_KEY:
        return {"status": "ERROR", "reason": "Gemini API key not configured", 
                "immediate_action": "Configure API key", "long_term_recommendation": "Set up environment variables"}
    
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
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=10)
        response.raise_for_status()
        
        ai_response = response.json()
        response_text = ai_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not response_text:
            raise ValueError("Empty response from Gemini")
        
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        decision = json.loads(response_text)
        
        return decision
        
    except Exception as e:
        print(f"AI Decision Error: {e}")
        return {
            "status": "ERROR",
            "reason": f"Failed to get AI decision: {str(e)}",
            "immediate_action": "Check API configuration",
            "long_term_recommendation": "Verify API keys and network connection"
        }

def display_ai_decision():
    """Display AI decision in UI"""
    decision = get_ai_decision()
    
    status_colors = {
        "GOOD": "#10b981",
        "WARNING": "#f59e0b",
        "CRITICAL": "#ef4444",
        "ERROR": "#6b7280"
    }
    
    color = status_colors.get(decision['status'], "#6b7280")
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}15, {color}25);
        border: 2px solid {color};
        padding: 30px;
        border-radius: 20px;
        margin-top: 20px;
    ">
        <div style="
            display: inline-block;
            background: {color};
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 20px;
        ">
            {decision['status']}
        </div>
        
        <h3 style="margin: 15px 0; color: #1e293b;">Analysis</h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 20px;">{decision['reason']}</p>
        
        <h3 style="margin: 15px 0; color: #1e293b;">⚡ Immediate Action</h3>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 20px;">{decision['immediate_action']}</p>
        
        <h3 style="margin: 15px 0; color: #1e293b;">🎯 Long-Term Recommendation</h3>
        <p style="color: #475569; line-height: 1.6;">{decision['long_term_recommendation']}</p>
    </div>
    """
    
    return html

# ---------------- EMAIL FUNCTION ----------------
def send_decision_email():
    """Send AI decision via email"""
    
    if not resend.api_key:
        return "❌ Error: Resend API key not configured"
    
    decision = get_ai_decision()
    
    try:
        params = {
            "from": f"AI Decision System <{RESEND_FROM_EMAIL}>",
            "to": ["pruturaj3003@gmail.com"],
            "subject": "🚨 AI Academic Risk Report – Action Required",
            "html": f"""
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #ef4444;">Status: {decision['status']}</h2>
                    <p><strong>Reason:</strong> {decision['reason']}</p>
                    <p><strong>Immediate Action:</strong> {decision['immediate_action']}</p>
                    <p><strong>Long-Term Recommendation:</strong> {decision['long_term_recommendation']}</p>
                    <hr style="border: 1px solid #e5e7eb; margin: 20px 0;" />
                    <p style="color: #6b7280; font-size: 14px;">
                        This decision was generated automatically by the AI Decision Assistant.
                    </p>
                </div>
            """
        }
        
        email = resend.Emails.send(params)
        return f"✅ Email sent successfully! Email ID: {email.get('id', 'N/A')}"
        
    except Exception as e:
        return f"❌ Error sending email: {str(e)}"

# ---------------- PREDICTION FUNCTION ----------------
def predict_delivery_time(email, age, rating, distance, order_type, vehicle_type):

    order_encoded = ORDER_MAP[order_type]
    vehicle_encoded = VEHICLE_MAP[vehicle_type]

    X = pd.DataFrame([[
        age,
        rating,
        distance,
        order_encoded,
        vehicle_encoded
    ]], columns=FEATURES)

    pred = model.predict(X)[0]

    card = f"""
    <div style="
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 40px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 25px 50px -12px rgba(139, 92, 246, 0.5);
        position: relative;
        overflow: hidden;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    ">
        <div style="
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            filter: blur(60px);
        "></div>
        
        <div style="position: relative; z-index: 2;">
            <div style="
                font-size: 14px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 2px;
                opacity: 0.9;
                margin-bottom: 8px;
            ">Estimated Delivery Time</div>
            
            <div style="
                font-size: 72px;
                font-weight: 900;
                margin: 20px 0;
                text-shadow: 0 10px 30px rgba(0,0,0,0.3);
                line-height: 1;
            ">
                {pred:.1f} <span style="font-size: 36px; opacity: 0.8;">mins</span>
            </div>
            
            <div style="
                background: rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                padding: 16px 20px;
                border-radius: 16px;
                font-size: 15px;
                margin-top: 24px;
                border: 1px solid rgba(255,255,255,0.2);
                display: flex;
                align-items: center;
                gap: 12px;
            ">
                <span style="font-size: 24px;">🚀</span>
                <span style="line-height: 1.5;">Higher-rated delivery partners and shorter distances reduce delivery time significantly.</span>
            </div>
        </div>
    </div>
    
    <style>
        @keyframes slideUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
    </style>
    """

    return card

# ---------------- PREMIUM CSS ----------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { 
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}

body, .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    padding: 40px 20px !important;
}

#main {
    max-width: 1000px;
    margin: auto;
    background: rgba(255, 255, 255, 0.98);
    padding: 60px 50px;
    border-radius: 32px;
    box-shadow: 
        0 0 0 1px rgba(255,255,255,0.1),
        0 50px 100px -20px rgba(0, 0, 0, 0.5),
        0 30px 60px -30px rgba(139, 92, 246, 0.4);
    position: relative;
    animation: fadeIn 0.8s ease-out;
}

#main::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #f59e0b);
    border-radius: 32px 32px 0 0;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

h1 {
    font-size: 64px !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 8px !important;
    letter-spacing: -2px;
    line-height: 1.1 !important;
}

.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 50px;
    font-size: 18px;
    font-weight: 500;
    letter-spacing: 0.3px;
}

.section-header {
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #6366f1;
    margin: 35px 0 20px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-header::before {
    content: '';
    width: 4px;
    height: 20px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 2px;
}

.gradio-textbox input,
.gradio-number input,
.gradio-dropdown select {
    border: 2px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
    background: white !important;
}

.gradio-textbox input:focus,
.gradio-number input:focus,
.gradio-dropdown select:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.1) !important;
    outline: none !important;
}

.gradio-slider input[type="range"] {
    height: 8px !important;
    border-radius: 4px !important;
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

.gradio-slider input[type="range"]::-webkit-slider-thumb {
    width: 24px !important;
    height: 24px !important;
    background: white !important;
    border: 3px solid #8b5cf6 !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4) !important;
}

.primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 18px 32px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 30px -10px rgba(139, 92, 246, 0.5) !important;
    text-transform: none !important;
    letter-spacing: 0.5px !important;
    margin-top: 20px !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 40px -10px rgba(139, 92, 246, 0.6) !important;
}

.primary:active {
    transform: translateY(0) !important;
}

.secondary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: white !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.5) !important;
}

.secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 40px -10px rgba(16, 185, 129, 0.6) !important;
}

label {
    font-weight: 600 !important;
    color: #334155 !important;
    font-size: 14px !important;
    margin-bottom: 8px !important;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 50px;
    padding-top: 30px;
    border-top: 1px solid #e2e8f0;
    font-weight: 500;
}

.footer-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 4px;
}

.gr-row {
    gap: 20px;
}
"""

# ---------------- UI ----------------
with gr.Blocks(css=css, title="🚀 Smart Delivery AI System") as app:

    with gr.Column(elem_id="main"):

        gr.Markdown("""
        <h1>Smart Delivery ETA</h1>
        <div class="subtitle">
        AI-powered food delivery time prediction using advanced Machine Learning
        </div>
        """, unsafe_allow_html=True)

        # Tab 1: Delivery Prediction
        with gr.Tab("🚀 Delivery Prediction"):
            
            gr.Markdown('<div class="section-header">📧 User Information</div>')
            email = gr.Textbox(
                label="Email Address", 
                placeholder="you@example.com",
                info="Enter your email to receive delivery updates"
            )

            gr.Markdown('<div class="section-header">🎯 Delivery Parameters</div>')

            with gr.Row():
                age = gr.Slider(
                    18, 60, 
                    value=30, 
                    label="Delivery Partner Age",
                    info="Age of the delivery person"
                )
                rating = gr.Slider(
                    1.0, 5.0, 
                    step=0.1, 
                    value=4.5, 
                    label="Delivery Partner Rating",
                    info="Customer rating (1-5 stars)"
                )

            distance = gr.Number(
                value=5.0, 
                label="Distance (km)",
                info="Distance from restaurant to delivery location"
            )

            with gr.Row():
                order_type = gr.Dropdown(
                    ["Buffet", "Drinks", "Meal", "Snack"],
                    label="Type of Order",
                    value="Meal",
                    info="Select the type of food order"
                )

                vehicle_type = gr.Dropdown(
                    ["Bicycle", "Electric Scooter", "Motorcycle", "Scooter"],
                    label="Type of Vehicle",
                    value="Motorcycle",
                    info="Delivery vehicle type"
                )

            predict_btn = gr.Button("🚀 Predict Delivery Time", variant="primary")
            output = gr.HTML()

            predict_btn.click(
                predict_delivery_time,
                inputs=[email, age, rating, distance, order_type, vehicle_type],
                outputs=output
            )
        
        # Tab 2: AI Decision System
        with gr.Tab("🤖 AI Decision System"):
            
            gr.Markdown('<div class="section-header">🧠 Academic Performance Analysis</div>')
            gr.Markdown("""
            <p style="color: #64748b; margin-bottom: 20px;">
            Get AI-powered insights on academic performance and receive actionable recommendations.
            </p>
            """, unsafe_allow_html=True)
            
            with gr.Row():
                analyze_btn = gr.Button("🧠 Analyze with AI", variant="primary", scale=2)
                email_btn = gr.Button("📧 Send Email Report", variant="secondary", scale=1)
            
            ai_output = gr.HTML()
            email_status = gr.Textbox(label="Email Status", interactive=False)
            
            analyze_btn.click(
                display_ai_decision,
                inputs=[],
                outputs=ai_output
            )
            
            email_btn.click(
                send_decision_email,
                inputs=[],
                outputs=email_status
            )

        gr.Markdown("""
        <div class="footer">
            Built with <span class="footer-badge">Gradient Boosting</span> 
            <span class="footer-badge">Gemini AI</span> 
            <span class="footer-badge">Resend Email</span> 
            <span class="footer-badge">Gradio UI</span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app.launch()