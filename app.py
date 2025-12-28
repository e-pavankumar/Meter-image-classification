import os
import shutil
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "meter_classifier.pth")

os.makedirs(UPLOADS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""

    if request.method == "POST":
        upload_type = request.form.get("uploadType")
        files = request.files.getlist("files")

        if not files or files[0].filename == "":
            return render_template("index.html", message=" No input selected")

      
        if upload_type == "image":
            file = files[0]
            filename = secure_filename(file.filename)
            base_name = os.path.splitext(filename)[0]

            base_folder = os.path.join(UPLOADS_DIR, base_name)
            clear_dir = os.path.join(base_folder, "classification", "clear")
            unclear_dir = os.path.join(base_folder, "classification", "unclear")

            os.makedirs(clear_dir, exist_ok=True)
            os.makedirs(unclear_dir, exist_ok=True)

            temp_path = os.path.join(base_folder, filename)
            file.save(temp_path)

            img = Image.open(temp_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(img_tensor), dim=1)
                pred = torch.argmax(probs, dim=1).item()

            target = clear_dir if pred == 0 else unclear_dir
            shutil.move(temp_path, os.path.join(target, filename))

            message = "Image classified successfully"

     
        else:
            root_folder = files[0].filename.split("/")[0]

            base_folder = os.path.join(UPLOADS_DIR, root_folder)
            clear_dir = os.path.join(base_folder, "classification", "clear")
            unclear_dir = os.path.join(base_folder, "classification", "unclear")

            os.makedirs(clear_dir, exist_ok=True)
            os.makedirs(unclear_dir, exist_ok=True)

            count = 0

            for file in files:
                parts = file.filename.split("/")

                if len(parts) != 2:
                    continue

                if not parts[1].lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                filename = secure_filename(parts[1])
                temp_path = os.path.join(base_folder, filename)
                file.save(temp_path)

                img = Image.open(temp_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = torch.softmax(model(img_tensor), dim=1)
                    pred = torch.argmax(probs, dim=1).item()

                target = clear_dir if pred == 0 else unclear_dir
                shutil.move(temp_path, os.path.join(target, filename))
                count += 1

            message = f"{count} images classified from folder"

    return render_template("index.html", message=message)
if __name__ == "__main__":
    app.run(debug=True)
