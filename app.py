from flask import Flask, request, jsonify
from pantry_system import PantrySearchSystem
import traceback

app = Flask(__name__)
system = None

def get_system():
    global system
    if system is None:
        system = PantrySearchSystem("pantry_data.json")
    return system

@app.route("/")
def home():
    return "Kansas Pantry Assistant is live."

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        print("🔹 /chat endpoint hit")
        data = request.get_json(force=True)
        print(f"🔹 Received data: {data}")
        query_text = data.get("text", "")
        if not query_text:
            return jsonify({"error": "No query text provided"}), 400
        response = get_system().process(query_text)
        return jsonify({"response": response})
    except Exception as e:
        print("Error in /chat:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
