from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_response, predict_class
import json

app = Flask(__name__)
CORS(app)
intents = json.loads(open('intents.json', 'r', encoding='utf-8').read())

#@app.get("/")
#def index_get():
#    return render_template("base.html")



@app.route("/predict", methods=['POST'])
#@cross_origin(origins=["https://chatbot-py-js.onrender.com"])
def predict():
    text = request.get_json().get("message")
    # TODO:check if text is valid
    ints = predict_class(text)
    response = get_response(ints, intents)
    message = {"answer":response}
    return jsonify(message)



if __name__ == "__main__":
    app.run(debug=True)
