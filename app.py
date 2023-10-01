from flask import Flask, render_template, url_for, request, session, redirect, Markup
from flask_pymongo import pymongo, MongoClient
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pickle
import pdfkit
import bcrypt

classifier = load_model('Trained_model.h5')
classifier._make_predict_function()

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)
app.secret_key = "testing"

#Connection With MongoDB Database
client = MongoClient("")
db = client["CropAdvisorAdmin"]
collection = db["AdminData"]
farmercollection = db["FarmerData"]

@app.route("/")
def index():
    if 'email' in session:
        return render_template("index.html")

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
        return redirect(url_for('index'))
    
    return render_template("login.html", error_message="Invalid Email/Password")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/farmer-registration', methods=['GET', 'POST'])
def farmerRegistration():
    if request.method == 'POST':
        
        full_name = request.form['fullName']
        gender = request.form['gender']
        mobile_number = request.form['mobileNumber']
        aadhar_number = request.form['aadhar']
        state = request.form['state']
        city = request.form['city']
        taluka = request.form['taluka']
        village = request.form['village']
        farm_address = request.form['address']
        farm_state = request.form['farm-state']
        farm_disrict = request.form['farm-district']
        farm_taluka = request.form['farm-taluka']
        farm_village = request.form['farm-village']
        pincode = request.form['pincode']
        survey_number= request.form['surveyNumber']
        land_in_acres = request.form['area']


        personal_details = {
            'full_name': full_name,
            'gender': gender,
            'mobile_number': mobile_number,
            'aadhar_number': aadhar_number,
            'state': state,
            'city': city,
            'taluka': taluka,
            'village': village
        }

        farm_details = {
            'farm_address': farm_address,
            'farm_state': farm_state,
            'farm_district': farm_disrict,
            'farm_taluka': farm_taluka,
            'farm_village': farm_village,
            'pincode': pincode,
            'survey_number': survey_number,
            'land_in_acres': land_in_acres
        }
        
#Check if farmer is already registered

        existing_farmer = farmercollection.find_one({
            '$or': [
                {'personal_details.full_name': full_name},
                {'personal_details.mobile_number': mobile_number},
                {'personal_details.aadhar_number': aadhar_number},
                {'farm_details.survey_number': survey_number},
            ]
        })

        if existing_farmer:
            # A farmer with the same field value already exists
            return render_template("farmer-registration.html", error="Farmer Already Exist!")

        farmercollection.insert_one({
    'personal_details': personal_details,
    'farm_details': farm_details
})
        
        # Stored some relevant information about farmer in the session
        session['full_name'] = full_name
        session['registered'] = True

        return redirect(url_for('farmerIndex'))  
    else:
        return render_template("farmer-registration.html")

@app.route('/farmer-index')
def farmerIndex():
    return render_template("farmerIndex.html")

@app.route("/logout_alt")
def logout_alt():
    session.clear()
    return redirect(url_for('adminIndex'))

@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)


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
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')

if __name__ == '__main__':
    app.run(debug=True)