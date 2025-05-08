from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("Received:", data)  # Debug: Log input
    return jsonify({"response": "✅ Backend is working fine and received your message."})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
