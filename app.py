from flask import Flask, request, jsonify
from pantry_system import PantrySearchSystem

app = Flask(__name__)
system = None

def get_system():
    global system
    if system is None:
        system = PantrySearchSystem("pantry_data.json")
    return system

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("text", "")
    response = get_system().process(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
