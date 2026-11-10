# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'AgriSense - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'AgriSense - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'AgriSense - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page



# render fertilizer recommendation result page



@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AgriSense - Crop Recommendation'

    try:
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        values = [N, P, K, temperature, humidity, ph, rainfall]

        if not all(np.isfinite(v) for v in values):
            raise ValueError("Please enter valid numerical values.")

        if any(v < 0 for v in values):
            raise ValueError("Values cannot be negative.")

        if humidity > 100:
            raise ValueError("Humidity must be between 0 and 100.")

        data = np.array([values], dtype=float)
        final_prediction = crop_recommendation_model.predict(data)[0]

        return render_template(
            'crop-result.html',
            prediction=final_prediction,
            title=title
        )

    except (ValueError, TypeError, KeyError) as e:
        return render_template(
            'crop.html',
            title=title,
            error=str(e)
        ), 400

    except Exception:
        app.logger.exception("Crop prediction failed")
        return render_template(
            'crop.html',
            title=title,
            error="Prediction could not be completed. Please try again."
        ), 500

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgriSense - Fertilizer Recommendation'

    try:
        crop_name = request.form['cropname']
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])

        if not all(np.isfinite(v) for v in [N, P, K]):
            raise ValueError("Please enter valid numerical values.")

        if any(v < 0 for v in [N, P, K]):
            raise ValueError("NPK values cannot be negative.")

        df = pd.read_csv('Data/fertilizer.csv')

        crop_data = df[df['Crop'].str.strip().str.lower() ==
                       crop_name.strip().lower()]

        if crop_data.empty:
            raise ValueError("Crop not found in fertilizer dataset.")

        nr = float(crop_data['N'].iloc[0])
        pr = float(crop_data['P'].iloc[0])
        kr = float(crop_data['K'].iloc[0])

        n = nr - N
        p = pr - P
        k = kr - K

        nutrient_gaps = {
            abs(n): ('N', n),
            abs(p): ('P', p),
            abs(k): ('K', k)
        }

        nutrient, gap = nutrient_gaps[max(nutrient_gaps)]

        if nutrient == 'N':
            key = 'NHigh' if gap < 0 else 'Nlow'
        elif nutrient == 'P':
            key = 'PHigh' if gap < 0 else 'Plow'
        else:
            key = 'KHigh' if gap < 0 else 'Klow'

        recommendation = Markup(str(fertilizer_dic[key]))

        return render_template(
            'fertilizer-result.html',
            recommendation=recommendation,
            title=title
        )

    except (ValueError, TypeError, KeyError) as e:
        return render_template(
            'fertilizer.html',
            title=title,
            error=str(e)
        ), 400

    except Exception:
        app.logger.exception("Fertilizer recommendation failed")
        return render_template(
            'fertilizer.html',
            title=title,
            error="Fertilizer recommendation could not be completed."
        ), 500


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgriSense - Disease Detection'

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template('disease.html', title=title)

        try:
            img = file.read()
            prediction = predict_image(img)
            prediction = Markup(str(disease_dic[prediction]))

            return render_template(
                'disease-result.html',
                prediction=prediction,
                title=title
            )
        except Exception as e:
            print("Disease prediction error:", e)
            return render_template('disease.html', title=title)

    return render_template('disease.html', title=title)

if __name__ == '__main__':
    app.run(debug=False)
