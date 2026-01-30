from flask import Flask, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)


# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET"])
def api_health():
    return jsonify({"Message": "API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            data["Avg. Session Length"],
            data["Time on App"],
            data["Time on Website"],
            data["Length of Membership"]
        ]

        prediction = model.predict([features])

        return jsonify({
            "Predicted Yearly Amount Spent": round(prediction[0], 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
