import os
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

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        query_text = data.get("text", "")
        response = get_system().process(query_text)
        return jsonify({"response": response})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
<<<<<<< HEAD
=======


>>>>>>> d612625
