from flask import Flask, request, render_template_string
from langchain_google_community.gmail.search import GmailSearch
from langchain_google_community.gmail.utils import get_gmail_credentials, build_resource_service


# Defining the path for the credentials
import os
credential_path = os.path.dirname(os.getcwd())
secret_path = os.path.join(credential_path, "credentials.json")



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    snippet = ""
    if request.method == "POST":
        sender = request.form["sender"]
        creds = get_gmail_credentials(
            token_file="token.json",
            client_secrets_file=secret_path,
            scopes=["https://www.googleapis.com/auth/gmail.readonly"]
        )
        resource = build_resource_service(credentials=creds)
        results = GmailSearch(api_resource=resource).run(f"from:{sender} label:unread")
        if results:
            result = f"📬 You have {len(results)} unread email(s) from {sender}."

            # Fetch the first message snippet
            first_message = results[0]
            
            # message = get_message_tool.run({"message_id": results[0]["id"]})

            # Extract snippet
            snippet = first_message.get("snippet", "[No preview available]")
        else:
            result = f"✅ No unread emails from {sender}."


    return render_template_string("""
        <html>
        <head>
            <title>Gmail Checker</title>
        </head>
        <body style="font-family: Arial, sans-serif; padding: 40px;">
            <h1>📧 Gmail Checker</h1>
            <form method="post">
                <label for="sender">Enter sender email address:</label><br>
                <input name="sender" type="email" required><br><br>
                <input type="submit" value="Check for Unread Emails">
            </form>
            <hr>
            {% if result %}
                <h3>{{ result }}</h3>
            {% endif %}
            {% if snippet %}
                <p><strong>First unread message preview:</strong><br>{{ snippet }}</p>
            {% endif %}
        </body>
        </html>
    """, result=result, snippet=snippet)

if __name__ == "__main__":
    app.run(debug=True)
