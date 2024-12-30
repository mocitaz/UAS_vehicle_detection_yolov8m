from flask import Flask, request, render_template
import os
from ultralytics import YOLO
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Path ke model YOLO
MODEL_PATH = '/Users/Luthfi/Documents/Collage/Subject/Semester 6/Visi Komputer/Tubes/Program Tubes/runs/detect/train10/weights/best.pt'
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")

# Load model YOLO
model = YOLO(MODEL_PATH)
print("Model Loaded:", model)

# Path untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = '/Users/Luthfi/Documents/Collage/Subject/Semester 6/Visi Komputer/Tubes/Program Tubes/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Pemetaan ulang label
label_mapping = {
    "bicycle": "motorcycle",  
    "car": "truck",           
    "truck": "car",           
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Sanitasi nama file
    file.filename = secure_filename(file.filename)
    
    # Cek apakah ekstensi file gambar valid
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return "Invalid file type. Only image files are allowed.", 400

    if file:
        # Simpan file yang diunggah
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"File Uploaded: {file_path}")

        # Deteksi objek menggunakan YOLO dengan parameter yang dioptimalkan
        results = model.predict(
            source=file_path, 
            conf=0.12,    
            iou=0.25,     
            imgsz=3072,  
        )


        # Periksa apakah ada deteksi
        if len(results[0].boxes) == 0:
            print("No objects detected!")
            return render_template('index.html', filename=None, detections=[])

        # Ambil data deteksi
        detections = results[0].boxes.xyxy.numpy()  # Bounding boxes
        classes = results[0].boxes.cls.numpy()     # Class IDs
        confidences = results[0].boxes.conf.numpy() # Confidence scores

        # Filter deteksi hanya untuk kelas 
        allowed_classes = ['car', 'truck', 'bus', 'bicycle'] 
        filtered_detections = [
            (box, cls, conf) for box, cls, conf in zip(detections, classes, confidences)
            if model.names[int(cls)] in allowed_classes
        ]

        if not filtered_detections:
            return render_template('index.html', filename=None, detections=[])

        detections, classes, confidences = zip(*filtered_detections)

        # Baca gambar asli
        image = cv2.imread(file_path)
        height, width, _ = image.shape

        # Gambar bounding box 
        detection_data = []
        for box, cls, conf in zip(detections, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            original_label = model.names[int(cls)]
            
            # Lakukan mapping label jika ada di `label_mapping`
            corrected_label = label_mapping.get(original_label, original_label)
            
            # Tambahkan data ke dalam daftar untuk ditampilkan
            detection_data.append((corrected_label, f"{conf:.2%}", (x1, y1, x2, y2)))

            # Gambar bounding box pada gambar
            label = f"{corrected_label}: {conf:.2%}"
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Simpan gambar dengan bounding box
        output_filename = 'detected_' + file.filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, image)

        # Validasi apakah gambar berhasil disimpan
        if os.path.exists(output_path):
            print(f"Gambar hasil deteksi berhasil disimpan di {output_path}")
        else:
            print("Error: Gambar hasil deteksi tidak tersimpan!")
            return "Error: Gambar tidak dapat disimpan.", 500

        return render_template('index.html', filename='uploads/' + output_filename, detections=detection_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
