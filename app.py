from flask import Flask, request, jsonify
from pantry_system import PantrySearchSystem
import traceback

app = Flask(__name__)
system = None

def get_system():
    global system
    if system is None:
        system = PantrySearchSystem("pantry_data.json")  # Confirm this file exists!
    return system

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        query = data.get("text", "")
        response = get_system().process(query)
        return jsonify({"response": response})
    except Exception as e:
        print("ERROR during /chat:", traceback.format_exc())  # log full stack trace
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)