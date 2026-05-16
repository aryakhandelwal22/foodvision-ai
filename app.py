from flask import Flask, render_template, request
import os
from datetime import datetime
from predict import predict_food

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

history = []

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        # Prediction
        food, calories, confidence, nutrition = predict_food(filepath)

        # BMI
        weight = request.form.get("weight")
        height = request.form.get("height")

        bmi = "Not Provided"
        recommendation = "Maintain balanced diet."

        if weight and height:

            weight = float(weight)
            height = float(height) / 100

            bmi_value = round(
                weight / (height * height),
                1
            )

            bmi = bmi_value

            if bmi_value < 18.5:
                recommendation = "Increase healthy calorie intake."

            elif bmi_value < 25:
                recommendation = "Great shape. Maintain current diet."

            else:
                recommendation = "Reduce high-calorie foods."

        result = {

            "food": food,
            "calories": calories,
            "confidence": confidence,
            "protein": nutrition["protein"],
            "carbs": nutrition["carbs"],
            "fat": nutrition["fat"],
            "health": nutrition["health"],
            "image": filepath,
            "bmi": bmi,
            "recommendation": recommendation
        }

        history.append({

            "food": food,
            "calories": calories,
            "time": datetime.now().strftime("%H:%M")
        })

    return render_template(
        "index.html",
        result=result,
        history=history
    )

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5000
    )