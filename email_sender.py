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

# Decision data structure for delivery performance
decision = {
    "status": "WARNING",
    "reason": "Overall delivery performance is moderate with an average delivery time of 27.3 minutes, but 8.2% of deliveries are delayed beyond the 30-minute target.",
    "immediate_action": "Assign high-rated delivery partners (4.5+ rating) to orders above 8km to reduce delays.",
    "long_term_recommendation": "Implement dynamic routing algorithm to optimize partner assignments based on real-time traffic and distance."
}

def send_email(decision_data=None):
    """Send email with AI decision report for delivery performance"""
    
    if decision_data is None:
        decision_data = decision
    
    try:
        params = {
            "from": f"Delivery AI System <{RESEND_FROM_EMAIL}>",
            "to": ["pruturaj3003@gmail.com"],
            "subject": "🚨 Delivery Performance Alert – Action Required",
            "html": f"""
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background: #f9fafb; border-radius: 12px;">
                    <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); padding: 30px; border-radius: 12px 12px 0 0; text-align: center;">
                        <h1 style="color: white; margin: 0; font-size: 28px;">🚀 Delivery Performance Report</h1>
                    </div>
                    
                    <div style="background: white; padding: 30px; border-radius: 0 0 12px 12px;">
                        <div style="background: #{"10b981" if decision_data['status'] == "GOOD" else "#f59e0b" if decision_data['status'] == "WARNING" else "#ef4444"}; color: white; padding: 10px 20px; border-radius: 8px; display: inline-block; font-weight: bold; margin-bottom: 20px;">
                            Status: {decision_data['status']}
                        </div>
                        
                        <h3 style="color: #1e293b; margin-top: 20px;">📊 Analysis</h3>
                        <p style="color: #475569; line-height: 1.6;">{decision_data['reason']}</p>
                        
                        <h3 style="color: #1e293b; margin-top: 25px;">⚡ Immediate Action</h3>
                        <p style="color: #475569; line-height: 1.6; background: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;">
                            {decision_data['immediate_action']}
                        </p>
                        
                        <h3 style="color: #1e293b; margin-top: 25px;">🎯 Long-Term Recommendation</h3>
                        <p style="color: #475569; line-height: 1.6; background: #dbeafe; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
                            {decision_data['long_term_recommendation']}
                        </p>
                        
                        <hr style="border: 1px solid #e5e7eb; margin: 30px 0;" />
                        
                        <p style="color: #6b7280; font-size: 14px; text-align: center;">
                            This report was generated automatically by the Delivery AI System.<br>
                            <strong>Powered by Gemini AI & Machine Learning</strong>
                        </p>
                    </div>
                </div>
            """
        }
        
        email = resend.Emails.send(params)
        print("✅ Email sent successfully!")
        print(f"📧 Email ID: {email.get('id', 'N/A')}")
        return email
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        raise

if __name__ == "__main__":
    try:
        send_email()
        print("\n🎉 Test email sent! Check your inbox.")
    except Exception as e:
        print(f"Error: {e}")