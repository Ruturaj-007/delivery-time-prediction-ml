import gradio as gr
import pandas as pd
import joblib
import os

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "delivery_time_model.joblib"))
FEATURES = joblib.load(os.path.join(BASE_DIR, "model_features.joblib"))

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
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 20px 50px rgba(102,126,234,0.45);
        animation: slideUp 0.6s ease-out;
    ">
        <div style="font-size:14px; opacity:0.85;">Estimated Delivery Time</div>
        <div style="font-size:56px; font-weight:900; margin:10px 0;">
            {pred:.1f} mins
        </div>
        <div style="
            background: rgba(255,255,255,0.15);
            padding: 14px;
            border-radius: 14px;
            font-size:15px;
            margin-top:15px;
        ">
            🚀 Higher-rated delivery partners and shorter distances reduce delivery time.
        </div>
    </div>
    """

    return card


# ---------------- CSS ----------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }

.gradio-container {
    background: radial-gradient(circle at top, #0f172a, #020617);
    min-height: 100vh;
}

#main {
    max-width: 950px;
    margin: auto;
    margin-top: 40px;
    background: rgba(255,255,255,0.96);
    padding: 45px;
    border-radius: 30px;
    box-shadow: 0 30px 80px rgba(0,0,0,0.35);
}

h1 {
    font-size: 52px;
    font-weight: 900;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

.subtitle {
    text-align:center;
    color:#475569;
    margin-bottom:35px;
    font-size:16px;
}

.primary {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border-radius: 16px !important;
    padding: 16px !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}
"""

# ---------------- UI ----------------
with gr.Blocks(css=css, title="Smart Delivery Time Predictor") as app:

    with gr.Column(elem_id="main"):

        gr.Markdown("""
        <h1>Smart Delivery ETA</h1>
        <div class="subtitle">
        AI-powered food delivery time prediction using Machine Learning
        </div>
        """, unsafe_allow_html=True)

        gr.Markdown("### User Details")
        email = gr.Textbox(label="Email Address", placeholder="you@example.com")

        gr.Markdown("### Delivery Parameters")

        with gr.Row():
            age = gr.Slider(18, 60, value=30, label="Delivery Partner Age")
            rating = gr.Slider(1.0, 5.0, step=0.1, value=4.5, label="Delivery Partner Rating")

        distance = gr.Number(value=5.0, label="Distance (km)")

        with gr.Row():
            order_type = gr.Dropdown(
                ["Buffet", "Drinks", "Meal", "Snack"],
                label="Type of Order",
                value="Meal"
            )

            vehicle_type = gr.Dropdown(
                ["Bicycle", "Electric Scooter", "Motorcycle", "Scooter"],
                label="Type of Vehicle",
                value="Motorcycle"
            )

        predict_btn = gr.Button("Predict Delivery Time", variant="primary")

        output = gr.HTML()

        predict_btn.click(
            predict_delivery_time,
            inputs=[email, age, rating, distance, order_type, vehicle_type],
            outputs=output
        )

        gr.Markdown("""
        <hr>
        <div style="text-align:center; color:#94a3b8; font-size:14px;">
        Built with Gradient Boosting • Feature Engineering • Gradio UI
        </div>
        """, unsafe_allow_html=True)

app.launch()
