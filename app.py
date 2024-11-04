from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_response, predict_class
import json
import os
import nltk

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins":"*", "methods":["POST"]}})
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

@app.get("/")
def index_get():
    return render_template("base.html")



@app.route("/api/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    # TODO:check if text is valid
    ints = predict_class(text)
    response = get_response(ints, intents)
    message = {"answer":response}
    return jsonify(message)



if __name__ == "__main__":
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
