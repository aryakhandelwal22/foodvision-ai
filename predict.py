import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import pandas as pd

# Load model
model = tf.keras.models.load_model(
    "models/food_model.h5"
)

# Load calorie data
calorie_df = pd.read_csv(
    "calorie_data.csv"
)

# Food classes
class_names = list(calorie_df['food'])

# Nutrition info
nutrition_data = {

    "pizza": {
        "protein": "11g",
        "carbs": "33g",
        "fat": "10g",
        "health": "Moderate"
    },

    "ice_cream": {
        "protein": "4g",
        "carbs": "24g",
        "fat": "7g",
        "health": "Treat Food"
    },

    "steak": {
        "protein": "25g",
        "carbs": "0g",
        "fat": "19g",
        "health": "High Protein"
    },

    "sushi": {
        "protein": "13g",
        "carbs": "28g",
        "fat": "3g",
        "health": "Healthy"
    }
}

def predict_food(img_path):

    img = image.load_img(
        img_path,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    calories = calorie_df[
        calorie_df['food'] == predicted_class
    ]['calories'].values[0]

    nutrition = nutrition_data.get(
        predicted_class,
        {
            "protein":"N/A",
            "carbs":"N/A",
            "fat":"N/A",
            "health":"Unknown"
        }
    )

    return (
        predicted_class,
        calories,
        confidence,
        nutrition
    )