"""
Elly Ai - AI Chatbot (Flask + Groq API)
College Project Demo

Setup:
    1. pip install -r requirements.txt
    2. Set your API key (PowerShell):  $env:GROQ_API_KEY="your-key-here"
       (Mac/Linux):                    export GROQ_API_KEY="your-key-here"
    3. python app.py
    4. Open http://127.0.0.1:5000 in your browser

Get a free API key at: https://console.groq.com/keys
"""

import os
from flask import Flask, render_template, request, jsonify
from groq import Groq

app = Flask(__name__)

# Load API key from environment variable (never hardcode your key)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"
SYSTEM_PROMPT = (
    "You are Elly AI, a friendly and helpful AI assistant built for a "
    "college project. Keep answers clear, concise, and helpful. "
    "Reply in plain text only: no markdown, no **bold**, no tables, "
    "no HTML tags like <br>. Use plain numbered or dashed lists with "
    "real line breaks instead.")


@app.route("/")
def home():
    """Render the chat UI."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Receive a user message + conversation history, return the AI reply."""
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])  # list of {"role": ..., "content": ...}

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Build the conversation for the API call
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": user_message}]
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000,
        )
        reply_text = response.choices[0].message.content
        return jsonify({"reply": reply_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
