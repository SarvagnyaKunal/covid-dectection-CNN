import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, request, render_template, redirect, url_for, jsonify, session
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
from PIL import Image
from datetime import datetime, timedelta
import os
import uuid
import bcrypt
from gridfs import GridFS

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey123!')
app.permanent_session_lifetime = timedelta(days=7)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# MongoDB setup
client = MongoClient(os.environ.get('MONGODB_URI', 'mongodb://localhost:27017/'))
db = client['Covidexpt']
users = db.users
reports = db.reports
fs = GridFS(db)

# COVIDNet model definition
class CovidNet(nn.Module):
    def __init__(self):
        super(CovidNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 26 * 26, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CovidNet().to(device)
model.load_state_dict(torch.load('model2.pth', map_location=device))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Helper functions
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Routes
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('home'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = users.find_one({'email': email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session.permanent = True
            session['user'] = str(user['_id'])
            return redirect(url_for('loader', redirect_url=url_for('home')))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


@app.route('/signup', methods=['POST'])
def signup():
    try:
        user_data = {
            'name': request.form.get('name'),
            'age': request.form.get('age'),
            'gender': request.form.get('gender'),
            'email': request.form.get('email'),
            'password': bcrypt.hashpw(request.form.get('password').encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        }

        if not all(user_data.values()):
            return render_template('login.html', error='All fields are required')

        if users.find_one({'email': user_data['email']}):
            return render_template('login.html', error='Email already exists')

        user_id = users.insert_one(user_data).inserted_id
        session.permanent = True
        session['user'] = str(user_id)
        return redirect(url_for('loader', redirect_url=url_for('home')))

    except Exception as e:
        return render_template('login.html', error=str(e))


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/loader')
def loader():
    return render_template('loader.html', redirect_url=request.args.get('redirect_url', url_for('home')))


@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))

    user = users.find_one({'_id': ObjectId(session['user'])})
    user_reports = reports.find({'user_id': session['user']}).sort('timestamp', -1)
    return render_template('home.html', username=user['name'], reports=user_reports)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'user' not in session:
            return jsonify({'status': 'error', 'message': 'Unauthorized'}), 401

        if 'xray' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        file = request.files['xray']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400

        # Save file
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image
        image = Image.open(filepath)
        image = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(image)

        probability = torch.sigmoid(output).item()
        result = "COVID Detected" if probability < 0.5 else "Normal"
        confidence = (1 - probability) * 100 if result == "COVID Detected" else probability * 100

        file_data = file.read()

        # Store image in GridFS
        file_id = fs.put(file_data, filename=filename, content_type=file.content_type)

        # Store report with GridFS reference
        report_id = reports.insert_one({
            'user_id': session['user'],
            'file_id': str(file_id),
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'timestamp': datetime.now()
        }).inserted_id

        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': f"{confidence:.2f}%",
            'report_id': str(report_id)
        })

    except Exception as e:
        app.logger.error(f'Prediction error: {str(e)}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)