import os
from flask import Flask, redirect, render_template, request, jsonify, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

FALLBACK_REMEDIES = [
    {'title': 'Apply fungicide', 'description': 'Use a copper-based spray every 7 days'},
    {'title': 'Prune affected leaves', 'description': 'Remove and bag all visibly diseased foliage immediately'},
    {'title': 'Improve drainage', 'description': 'Reduce standing moisture and avoid wetting leaves when watering'}
]

REMEDIES = {
    'Tomato___Late_blight': [
        {'title': 'Apply copper fungicide', 'description': 'Spray copper-based fungicide every 7 days during wet weather'},
        {'title': 'Remove infected tissue', 'description': 'Cut and bag all blighted leaves and stems immediately'},
        {'title': 'Improve air circulation', 'description': 'Space plants wider and avoid overhead watering'}
    ],
    'Tomato___Early_blight': [
        {'title': 'Rotate crops', 'description': 'Do not plant tomatoes or potatoes in the same spot for 2-3 years'},
        {'title': 'Water at base', 'description': 'Use drip irrigation to keep foliage dry and prevent spore germination'},
        {'title': 'Apply mulch', 'description': 'Cover soil to prevent splashing of fungal spores onto lower leaves'}
    ],
    'Tomato___Bacterial_spot': [
        {'title': 'Apply bactericide', 'description': 'Use copper sprays combined with mancozeb for disease suppression'},
        {'title': 'Avoid overhead watering', 'description': 'Keep leaves as dry as possible to prevent bacterial spread'},
        {'title': 'Use disease-free seed', 'description': 'Ensure seeds and transplants are certified disease-free'}
    ],
    'Potato___Early_blight': [
        {'title': 'Apply protectant fungicides', 'description': 'Start spraying early in the season before symptoms appear'},
        {'title': 'Maintain plant vigor', 'description': 'Ensure proper fertilization, especially adequate nitrogen and phosphorus'},
        {'title': 'Destroy cull piles', 'description': 'Eliminate potato cull piles and volunteer potatoes that harbor the fungus'}
    ],
    'Potato___Late_blight': [
        {'title': 'Apply systemic fungicides', 'description': 'Use targeted fungicides immediately upon detection'},
        {'title': 'Destroy infected vines', 'description': 'Kill infected vines days before harvest to protect tubers'},
        {'title': 'Proper storage', 'description': 'Store tubers in cool, dry areas with good ventilation'}
    ],
    'Corn_(maize)__Common_rust': [
        {'title': 'Plant resistant hybrids', 'description': 'Select rust-resistant corn varieties for planting'},
        {'title': 'Foliar fungicides', 'description': 'Apply fungicides if rust pustules cover significant leaf area early in season'},
        {'title': 'Monitor crops closely', 'description': 'Scout fields regularly, especially during cool, humid weather'}
    ],
    'Apple___Apple_scab': [
        {'title': 'Rake up fallen leaves', 'description': 'Remove autumn foliage to reduce overwintering fungal spores'},
        {'title': 'Fungicide program', 'description': 'Apply preventative sprays from bud break until petal fall'},
        {'title': 'Prune for airflow', 'description': 'Thin the canopy to promote rapid drying of leaves after rain'}
    ],
    'Grape___Black_rot': [
        {'title': 'Remove mummies', 'description': 'Collect and destroy shriveled, infected grapes from vines and ground'},
        {'title': 'Apply fungicides', 'description': 'Spray early in the season, particularly from pre-bloom to 4 weeks post-bloom'},
        {'title': 'Canopy management', 'description': 'Prune and train vines to increase air circulation and sun exposure'}
    ]
}
class TempModel(nn.Module):
    def __init__(self):
        super(TempModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3, 3))

    def forward(self, inp):
        return self.conv1(inp)

model = TempModel()
model.load_state_dict(torch.load("ResNet50.pt"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict():
    try:
        # Get the uploaded image
        image = request.files['image']
        img = Image.open(image)
        img = transform(img)

        # Make a prediction
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            predicted_class = torch.argmax(output)

        return jsonify({'prediction': predicted_class.item()})

    except Exception as e:
        return jsonify({'error': str(e)})

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scans.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease_name = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    severity = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')
    #comment

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = predict(file_path)  # Call the predict function here
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        confidence = 92.5
        severity = 'High' if float(confidence) > 85 else 'Medium' if float(confidence) > 60 else 'Low'
        db.session.add(Scan(disease_name=title, confidence=float(confidence), severity=severity))
        db.session.commit()
        
        remedies = REMEDIES.get(title, FALLBACK_REMEDIES)
        
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name, simage=supplement_image_url, buy_link=supplement_buy_link, remedies=remedies)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), buy=list(supplement_info['buy link']))

@app.route('/history')
def history():
    scans = Scan.query.order_by(Scan.timestamp.desc()).limit(30).all()
    return render_template('history.html', scans=scans)

@app.route('/history/clear', methods=['POST'])
def clear_history():
    Scan.query.delete()
    db.session.commit()
    return redirect(url_for('history'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=False)
