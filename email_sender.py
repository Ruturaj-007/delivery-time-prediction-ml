import os
import resend
from dotenv import load_dotenv

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL")

if not resend.api_key:
    raise ValueError("RESEND_API_KEY not configured in .env file")

if not RESEND_FROM_EMAIL:
    raise ValueError("RESEND_FROM_EMAIL not configured in .env file")

# Decision data structure
decision = {
    "status": "WARNING",
    "reason": "Overall academic performance is moderate with an average score of 67.71, but 12.4% of students are at academic risk, particularly in mathematics.",
    "immediate_action": "Launch targeted small-group intervention sessions for students identified as at-risk in math.",
    "long_term_recommendation": "Introduce a structured, school-wide test preparation program to improve consistency and outcomes."
}

def send_email(decision_data=None):
    """Send email with AI decision report"""
    
    if decision_data is None:
        decision_data = decision
    
    try:
        params = {
            "from": f"AI Decision System <{RESEND_FROM_EMAIL}>",
            "to": ["pruturaj3003@gmail.com"],  # Replace with your email
            "subject": "🚨 AI Academic Risk Report – Action Required",
            "html": f"""
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <h2 style="color: #ef4444;">Status: {decision_data['status']}</h2>
                    <p><strong>Reason:</strong> {decision_data['reason']}</p>
                    <p><strong>Immediate Action:</strong> {decision_data['immediate_action']}</p>
                    <p><strong>Long-Term Recommendation:</strong> {decision_data['long_term_recommendation']}</p>
                    <hr style="border: 1px solid #e5e7eb; margin: 20px 0;" />
                    <p style="color: #6b7280; font-size: 14px;">
                        This decision was generated automatically by the AI Decision Assistant.
                    </p>
                </div>
            """
        }
        
        email = resend.Emails.send(params)
        print("Email sent successfully!")
        print(f"Email ID: {email}")
        return email
        
    except Exception as e:
        print(f"Error sending email: {e}")
        raise

if __name__ == "__main__":
    try:
        send_email()
    except Exception as e:
        print(f"Error: {e}")