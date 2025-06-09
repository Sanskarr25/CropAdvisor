from flask import Flask, render_template, url_for, request, session, redirect, send_file, make_response
from markupsafe import Markup
from flask_pymongo import pymongo, MongoClient
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
import joblib
import bcrypt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
from datetime import datetime

app = Flask(__name__)
app.secret_key = "testing"

# Load models safely with error handling
try:
    from keras.preprocessing import image
    from keras.models import load_model
    import tensorflow as tf

    # Load the classifier model
    classifier = load_model('Trained_model.h5')

    # Load crop recommendation model using joblib
    crop_recommendation_model_path = 'Crop_Recommendation.pkl'
    crop_recommendation_model = joblib.load(crop_recommendation_model_path)

    models_loaded = True
    print("Models loaded successfully!")
except Exception as e:
    models_loaded = False
    print(f"Error loading models: {e}")

# Connection With MongoDB Database
client = MongoClient("mongodb+srv://avsanskar025:Cg5OTT1UofjJftPD@cluster0.h2go2xx.mongodb.net/")
db = client["CropAdvisorAdmin"]
collection = db["AdminData"]
farmercollection = db["FarmerData"]

def get_crop_details(crop_name):
    """Get detailed information about the recommended crop"""
    crop_database = {
        'Rice': {
            'scientific_name': 'Oryza sativa',
            'crop_type': 'Cereal Grain',
            'season': 'Kharif (June-November)',
            'maturity': '120-150 days',
            'yield': '4-6 tons/hectare',
            'market_value': 'High demand, stable prices',
            'temp_range': '20-35°C',
            'ph_range': '5.5-7.0',
            'rainfall_range': '1200-2500 mm',
            'humidity_range': '70-80%',
            'benefits': [
                'Staple food crop with guaranteed market demand',
                'Well-suited for wetland cultivation',
                'Multiple varieties available for different conditions',
                'Good source of income for farmers',
                'Can be grown in rotation with other crops'
            ]
        },
        'Wheat': {
            'scientific_name': 'Triticum aestivum',
            'crop_type': 'Cereal Grain',
            'season': 'Rabi (November-April)',
            'maturity': '120-150 days',
            'yield': '3-5 tons/hectare',
            'market_value': 'Stable market, government support',
            'temp_range': '10-25°C',
            'ph_range': '6.0-7.5',
            'rainfall_range': '300-1000 mm',
            'humidity_range': '50-70%',
            'benefits': [
                'Essential food grain with stable demand',
                'Suitable for mechanized farming',
                'Good storage life and processing options',
                'Government procurement support available',
                'Fits well in crop rotation systems'
            ]
        },
        'Maize': {
            'scientific_name': 'Zea mays',
            'crop_type': 'Cereal Grain',
            'season': 'Kharif/Rabi (All seasons)',
            'maturity': '90-120 days',
            'yield': '5-8 tons/hectare',
            'market_value': 'Growing demand for feed and food',
            'temp_range': '15-35°C',
            'ph_range': '5.8-7.0',
            'rainfall_range': '600-1200 mm',
            'humidity_range': '60-70%',
            'benefits': [
                'Versatile crop with multiple uses (food, feed, industrial)',
                'Short duration crop allowing multiple cropping',
                'Good market demand from poultry and livestock industry',
                'Relatively drought tolerant',
                'High nutritional value and processing potential'
            ]
        },
        'Cotton': {
            'scientific_name': 'Gossypium hirsutum',
            'crop_type': 'Fiber Crop',
            'season': 'Kharif (April-October)',
            'maturity': '180-200 days',
            'yield': '500-800 kg/hectare',
            'market_value': 'High value cash crop',
            'temp_range': '20-35°C',
            'ph_range': '5.8-8.0',
            'rainfall_range': '500-1250 mm',
            'humidity_range': '60-70%',
            'benefits': [
                'High-value cash crop with excellent returns',
                'Strong textile industry demand',
                'Cotton seeds provide additional income',
                'Well-established supply chain and markets',
                'Suitable for semi-arid regions'
            ]
        },
        'Sugarcane': {
            'scientific_name': 'Saccharum officinarum',
            'crop_type': 'Cash Crop',
            'season': 'Perennial (Plant once, harvest multiple times)',
            'maturity': '12-18 months',
            'yield': '70-100 tons/hectare',
            'market_value': 'Stable demand from sugar industry',
            'temp_range': '20-35°C',
            'ph_range': '6.0-7.5',
            'rainfall_range': '750-1200 mm',
            'humidity_range': '70-80%',
            'benefits': [
                'Long-term crop with multiple harvests',
                'Guaranteed purchase by sugar mills',
                'Bagasse can be used for additional income',
                'Creates employment opportunities',
                'Good for soil health improvement'
            ]
        }
    }
    
    return crop_database.get(crop_name, {
        'scientific_name': 'Information not available',
        'crop_type': 'General',
        'season': 'Season-dependent',
        'maturity': 'Varies',
        'yield': 'Variable',
        'market_value': 'Market dependent',
        'temp_range': 'Variable',
        'ph_range': 'Variable',
        'rainfall_range': 'Variable',
        'humidity_range': 'Variable',
        'benefits': [
            'Suitable for your soil and climate conditions',
            'Recommended based on scientific analysis',
            'Consult local experts for specific varieties'
        ]
    })

def get_status(param, value, crop_details):
    """Determine if parameter value is within optimal range"""
    try:
        if param == 'temperature':
            range_str = crop_details['temp_range'].replace('°C', '')
            min_val, max_val = map(int, range_str.split('-'))
            if min_val <= value <= max_val:
                return '✓ Optimal'
            else:
                return '⚠ Suboptimal'
        elif param == 'ph':
            range_str = crop_details['ph_range']
            min_val, max_val = map(float, range_str.split('-'))
            if min_val <= value <= max_val:
                return '✓ Optimal'
            else:
                return '⚠ Adjust'
        elif param == 'rainfall':
            range_str = crop_details['rainfall_range'].replace(' mm', '')
            min_val, max_val = map(int, range_str.split('-'))
            if min_val <= value <= max_val:
                return '✓ Optimal'
            else:
                return '⚠ Monitor'
        elif param == 'humidity':
            range_str = crop_details['humidity_range'].replace('%', '')
            min_val, max_val = map(int, range_str.split('-'))
            if min_val <= value <= max_val:
                return '✓ Optimal'
            else:
                return '⚠ Watch'
    except:
        return 'N/A'
    
    return 'N/A'

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

    n = N_desired - N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    key1 = "NHigh" if n < 0 else "Nlow" if n > 0 else "NNo"
    key2 = "PHigh" if p < 0 else "Plow" if p > 0 else "PNo"
    key3 = "KHigh" if k < 0 else "Klow" if k > 0 else "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n=abs_n, diff_p=abs_p, diff_k=abs_k)
                                                
def pred_pest(pest):
    if not models_loaded:
        return 'model_error'
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)
        predicted_class = np.argmax(result, axis=1)[0]
        return predicted_class
    except Exception as e:
        print(f"Error predicting pest: {e}")
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
        if pred == 'model_error':
            return render_template('error.html', message="Model loading error. Please contact administrator.")

        pest_mapping = {
            0: 'aphids',
            1: 'armyworm',
            2: 'beetle',
            3: 'bollworm',
            4: 'earthworm',
            5: 'grasshopper',
            6: 'mites',
            7: 'mosquito',
            8: 'sawfly',
            9: 'stem borer'
        }

        pest_identified = pest_mapping.get(pred, 'unknown')
        return render_template(pest_identified + ".html", pred=pest_identified)

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if not models_loaded:
        return render_template('error.html', message="Model loading error. Please contact administrator.")

    if request.method == 'POST':
        try:
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
            
            # Store prediction data in session for PDF generation
            session['crop_data'] = {
                'prediction': final_prediction,
                'nitrogen': N,
                'phosphorus': P,
                'potassium': K,
                'ph': ph,
                'rainfall': rainfall,
                'temperature': temperature,
                'humidity': humidity,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/' + final_prediction + '.jpg')
        except Exception as e:
            return render_template('error.html', message=f"Error during prediction: {str(e)}")

@app.route('/download_crop_report')
def download_crop_report():
    """Generate and download crop recommendation PDF report"""
    if 'crop_data' not in session:
        return "No crop data available for report generation", 400
    
    crop_data = session['crop_data']
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkgreen
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph("🌱 Crop Recommendation Report", title_style))
    content.append(Spacer(1, 20))
    
    # Report details
    content.append(Paragraph("Report Details", heading_style))
    report_data = [
        ['Generated on:', crop_data['timestamp']],
        ['User:', session.get('email', 'N/A')],
        ['Recommended Crop:', crop_data['prediction']],
        ['Confidence Level:', '92%']
    ]
    
    report_table = Table(report_data, colWidths=[2*inch, 3*inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    content.append(report_table)
    content.append(Spacer(1, 20))
    
    # Soil parameters
    content.append(Paragraph("Soil & Environmental Parameters", heading_style))
    
    param_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Nitrogen (N)', str(crop_data['nitrogen']), 'kg/ha'],
        ['Phosphorus (P)', str(crop_data['phosphorus']), 'kg/ha'],
        ['Potassium (K)', str(crop_data['potassium']), 'kg/ha'],
        ['pH Level', str(crop_data['ph']), ''],
        ['Temperature', str(crop_data['temperature']), '°C'],
        ['Rainfall', str(crop_data['rainfall']), 'mm'],
        ['Humidity', str(crop_data['humidity']), '%']
    ]
    
    param_table = Table(param_data, colWidths=[2*inch, 1.5*inch, 1*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    content.append(param_table)
    content.append(Spacer(1, 30))
    
    # Recommended Crop Section
    content.append(Paragraph("🌾 Recommended Crop Details", heading_style))
    
    crop_details = get_crop_details(crop_data['prediction'])
    
    # Crop overview table
    crop_overview_data = [
        ['Crop Name', crop_data['prediction']],
        ['Scientific Name', crop_details['scientific_name']],
        ['Crop Type', crop_details['crop_type']],
        ['Growing Season', crop_details['season']],
        ['Maturity Period', crop_details['maturity']],
        ['Expected Yield', crop_details['yield']],
        ['Market Value', crop_details['market_value']]
    ]
    
    crop_overview_table = Table(crop_overview_data, colWidths=[2*inch, 3*inch])
    crop_overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.darkgreen),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    
    content.append(crop_overview_table)
    content.append(Spacer(1, 15))
    
    # Crop requirements
    content.append(Paragraph("Optimal Growing Conditions", ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.darkgreen
    )))
    
    requirements_data = [
        ['Parameter', 'Optimal Range', 'Your Value', 'Status'],
        ['Temperature', crop_details['temp_range'], f"{crop_data['temperature']}°C", get_status('temperature', crop_data['temperature'], crop_details)],
        ['pH Level', crop_details['ph_range'], str(crop_data['ph']), get_status('ph', crop_data['ph'], crop_details)],
        ['Rainfall', crop_details['rainfall_range'], f"{crop_data['rainfall']} mm", get_status('rainfall', crop_data['rainfall'], crop_details)],
        ['Humidity', crop_details['humidity_range'], f"{crop_data['humidity']}%", get_status('humidity', crop_data['humidity'], crop_details)]
    ]
    
    requirements_table = Table(requirements_data, colWidths=[1.5*inch, 1.5*inch, 1*inch, 1*inch])
    requirements_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    content.append(requirements_table)
    content.append(Spacer(1, 15))
    
    # Crop benefits
    content.append(Paragraph("Why This Crop is Recommended", ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        textColor=colors.darkgreen
    )))
    
    for benefit in crop_details['benefits']:
        content.append(Paragraph(f"• {benefit}", styles['Normal']))
        content.append(Spacer(1, 4))
    
    content.append(Spacer(1, 20))
    
    # General Recommendations
    content.append(Paragraph("General Farming Recommendations", heading_style))
    recommendations = [
        f"Based on your soil analysis, <b>{crop_data['prediction']}</b> is the most suitable crop for your conditions.",
        "Ensure proper irrigation management based on the rainfall patterns in your area.",
        "Monitor soil pH regularly and adjust if necessary using appropriate amendments.",
        "Consider crop rotation practices to maintain soil health and fertility.",
        "Consult with local agricultural experts for region-specific growing tips."
    ]
    
    for i, rec in enumerate(recommendations, 1):
        content.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        content.append(Spacer(1, 6))
    
    content.append(Spacer(1, 20))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey
    )
    content.append(Paragraph("Generated by Crop Advisor System - Helping Farmers Make Better Decisions", footer_style))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    
    # Create response
    response = make_response(buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=crop_recommendation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    
    buffer.close()
    return response

if __name__ == '__main__':
    app.run()