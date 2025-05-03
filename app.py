from flask import Flask, render_template, url_for, request, session, redirect
from markupsafe import Markup
from flask_pymongo import pymongo, MongoClient
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pickle
import bcrypt

classifier = load_model('Trained_model.h5')

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)
app.secret_key = "testing"

# MongoDB connection
client = MongoClient("mongodb+srv://avsanskar025:Cg5OTT1UofjJftPD@cluster0.h2go2xx.mongodb.net/")
db = client["CropAdvisorAdmin"]
collection = db["AdminData"]
farmercollection = db["FarmerData"]

@app.route("/")
def index():
    if 'email' in session:
        return redirect(url_for('farmerIndex'))
    return render_template("login.html")

@app.route('/index')
def adminIndex():
    return render_template("index.html")

@app.route("/login", methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']
    login_user = collection.find_one({'email': email, 'password': password})

    if login_user:
        session['email'] = email
        return redirect(url_for('farmerIndex'))

    return render_template("login.html", error_message="Invalid Email/Password")

@app.route("/logout")
@app.route("/logout_alt")
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/farmer-index')
def farmerIndex():
    if 'email' not in session:
        return redirect(url_for('index'))
    return render_template("farmerIndex.html")

@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():
    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')
    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n, p, k = N_desired - N_filled, P_desired - P_filled, K_desired - K_filled

    key1 = "NHigh" if n < 0 else "Nlow" if n > 0 else "NNo"
    key2 = "PHigh" if p < 0 else "Plow" if p > 0 else "PNo"
    key3 = "KHigh" if k < 0 else "Klow" if k > 0 else "KNo"

    return render_template(
        'Fertilizer-Result.html',
        recommendation1=Markup(fertilizer_dict[key1]),
        recommendation2=Markup(fertilizer_dict[key2]),
        recommendation3=Markup(fertilizer_dict[key3]),
        diff_n=abs(n),
        diff_p=abs(p),
        diff_k=abs(k)
    )

def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict_classes(test_image)
        return result
    except:
        return 'x'

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
        if pred == 'x':
            return render_template('unaptfile.html')

        pest_labels = [
            'aphids', 'armyworm', 'beetle', 'bollworm', 'earthworm',
            'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem borer'
        ]
        pest_identified = pest_labels[pred[0]]

        return render_template(f"{pest_identified}.html", pred=pest_identified)

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['potassium'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = crop_recommendation_model.predict(data)[0]

    return render_template('crop-result.html', prediction=prediction, pred='img/crop/' + prediction + '.jpg')
