import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename
from PIL import Image

# Flask App
app = Flask(__name__)
app.secret_key = "your_secret_key"

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/covid_db"  # Update with your MongoDB URI
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming two classes: 'Normal' and 'COVID'
model.load_state_dict(torch.load("covid_xray_model.pth", map_location=device))
model.eval()
model.to(device)

# Image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Routes

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        data = request.form
        name, age, gender, email, password = data["name"], data["age"], data["gender"], data["email"], data["password"]
        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

        if mongo.db.users.find_one({"email": email}):
            return "Email already exists! Try logging in."

        mongo.db.users.insert_one({"name": name, "age": age, "gender": gender, "email": email, "password": hashed_pw})
        return redirect(url_for("login"))
    
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email, password = request.form["email"], request.form["password"]
        user = mongo.db.users.find_one({"email": email})

        if user and bcrypt.check_password_hash(user["password"], password):
            session["email"] = email
            return redirect(url_for("home"))
        return "Invalid Credentials!"
    
    return render_template("login.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if "email" not in session:
        return redirect(url_for("login"))

    # Fetch user details from MongoDB
    user = mongo.db.users.find_one({"email": session["email"]})

    if request.method == "POST":
        if "xray" not in request.files:
            return "No file uploaded"

        file = request.files["xray"]
        filename = secure_filename(file.filename)
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        image = preprocess_image(filepath)
        output = model(image)
        _, predicted = torch.max(output, 1)

        result = "COVID" if predicted.item() == 1 else "Normal"

        os.remove(filepath)  # Clean up uploaded file
        return jsonify({"prediction": result})

    return render_template("home.html", name=user["name"], age=user["age"], gender=user["gender"])


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
