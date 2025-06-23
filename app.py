# Solution 1: Lazy Loading with Memory Management
from flask import Flask, render_template, url_for, request, session, redirect, send_file, make_response
from markupsafe import Markup
from flask_pymongo import pymongo, MongoClient
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
import joblib
import bcrypt
import gc  # Garbage collection for memory management
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = "testing"

# Global variables for models - will be loaded on demand
classifier = None
crop_recommendation_model = None
models_loaded = False

def load_models_on_demand():
    """Load models only when needed and optimize memory usage"""
    global classifier, crop_recommendation_model, models_loaded
    
    if not models_loaded:
        try:
            # Import TensorFlow with memory optimization
            import tensorflow as tf
            
            # Configure TensorFlow for memory efficiency
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True) if tf.config.list_physical_devices('GPU') else None
            
            # Load models
            from keras.preprocessing import image
            from keras.models import load_model
            
            # Load classifier with memory optimization
            classifier = load_model('Trained_model.h5')
            
            # Load crop recommendation model
            crop_recommendation_model_path = 'Crop_Recommendation.pkl'
            crop_recommendation_model = joblib.load(crop_recommendation_model_path)
            
            models_loaded = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    return True

def unload_models():
    """Unload models to free memory"""
    global classifier, crop_recommendation_model, models_loaded
    
    if models_loaded:
        del classifier
        del crop_recommendation_model
        classifier = None
        crop_recommendation_model = None
        models_loaded = False
        gc.collect()  # Force garbage collection
        print("Models unloaded to free memory")

# MongoDB connection (unchanged)
client = MongoClient("mongodb+srv://avsanskar025:Cg5OTT1UofjJftPD@cluster0.h2go2xx.mongodb.net/")
db = client["CropAdvisorAdmin"]
collection = db["AdminData"]
farmercollection = db["FarmerData"]

# Your existing helper functions (unchanged)
def get_crop_details(crop_name):
    # ... (keep your existing implementation)
    pass

def get_status(param, value, crop_details):
    # ... (keep your existing implementation)
    pass

def get_pest_details(pest_name):
    # ... (keep your existing implementation)
    pass

# Optimized prediction function
def pred_pest(pest):
    """Optimized pest prediction with memory management"""
    global classifier
    
    # Load models on demand
    if not load_models_on_demand():
        return 'model_error'
    
    try:
        from keras.preprocessing import image
        
        # Load and preprocess image
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make prediction
        result = classifier.predict(test_image)
        predicted_class = np.argmax(result, axis=1)[0]
        
        # Clean up variables
        del test_image, result
        gc.collect()
        
        return predicted_class
        
    except Exception as e:
        print(f"Error predicting pest: {e}")
        return 'x'

# Your existing routes (unchanged)
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
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/farmer-index')
def farmerIndex():
    if 'email' not in session:
        return redirect(url_for('index'))
    return render_template("farmerIndex.html")

@app.route("/logout_alt")
def logout_alt():
    session.clear()
    return redirect(url_for('index'))

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

# Optimized prediction route
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['image']
            filename = file.filename
            
            file_path = os.path.join('static/user uploaded', filename)
            file.save(file_path)
            
            # Make prediction
            pred = pred_pest(pest=file_path)
            
            # Clean up uploaded file to save space
            try:
                os.remove(file_path)
            except:
                pass
            
            if pred == 'x':
                return render_template('unaptfile.html')
            if pred == 'model_error':
                return render_template('error.html', message="Model loading error. Please contact administrator.")
            
            pest_mapping = {
                0: 'aphids', 1: 'armyworm', 2: 'beetle', 3: 'bollworm', 4: 'earthworm',
                5: 'grasshopper', 6: 'mites', 7: 'mosquito', 8: 'sawfly', 9: 'stem_borer'
            }
            
            pest_identified = pest_mapping.get(pred, 'unknown')
            pest_details = get_pest_details(pest_identified)
            
            return render_template('pest_result.html', 
                                 pest_name=pest_identified,
                                 pest_details=pest_details)
                                 
        except Exception as e:
            return render_template('error.html', message=f"Error during prediction: {str(e)}")

# Optimized crop prediction route
@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        try:
            # Load models on demand
            if not load_models_on_demand():
                return render_template('error.html', message="Model loading error. Please contact administrator.")
            
            N = int(request.form['nitrogen'])
            P = int(request.form['phosphorus'])
            K = int(request.form['potassium'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            final_prediction = final_prediction.capitalize()
            
            # Clean up variables
            del data, my_prediction
            gc.collect()
            
            # Store prediction data in session for PDF generation
            session['crop_data'] = {
                'prediction': final_prediction,
                'nitrogen': N, 'phosphorus': P, 'potassium': K,
                'ph': ph, 'rainfall': rainfall, 'temperature': temperature,
                'humidity': humidity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return render_template('crop-result.html', 
                                 prediction=final_prediction, 
                                 pred='img/crop/' + final_prediction + '.jpg')
                                 
        except Exception as e:
            return render_template('error.html', message=f"Error during prediction: {str(e)}")

# Your existing fertilizer and PDF routes (unchanged)
@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():
    # ... (keep your existing implementation)
    pass

@app.route('/download_crop_report')
def download_crop_report():
    # ... (keep your existing implementation)
    pass

if __name__ == '__main__':
    app.run()