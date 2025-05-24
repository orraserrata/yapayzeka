import os
import sys
from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import tensorflow as tf
import joblib
from data_preprocessing import extract_features
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Get the absolute path of the current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {CURRENT_DIR}")

# List all files in the current directory
print("\nFiles in current directory:")
for file in os.listdir(CURRENT_DIR):
    print(f"- {file}")

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load models with error handling
try:
    # Model dosyaları kök dizinde
    svm_model = joblib.load('svm_model.joblib')
    rf_model = joblib.load('rf_model.joblib')
    deep_learning_model = tf.keras.models.load_model('deep_learning_model.h5', compile=False)
    deep_learning_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Load class indices and create new label encoder
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Create new label encoder with current classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(list(class_indices.keys()))
    
    print("Tüm modeller başarıyla yüklendi.")
    print(f"Mevcut sınıf sayısı: {len(label_encoder.classes_)}")
    print("Sınıflar:", label_encoder.classes_)
except Exception as e:
    print(f"Model yükleme hatası: {str(e)}")
    print(f"Mevcut dizin: {os.getcwd()}")
    print(f"Dizin içeriği: {os.listdir('.')}")
    raise

def create_prediction_chart(predictions):
    """Tahmin sonuçları için çubuk grafik oluşturur"""
    plt.figure(figsize=(10, 6))
    models = list(predictions.keys())
    confidences = [pred['confidence'] for pred in predictions.values()]
    
    bars = plt.bar(models, confidences)
    plt.title('Model Tahmin Güven Oranları')
    plt.xlabel('Modeller')
    plt.ylabel('Güven Oranı (%)')
    plt.ylim(0, 100)
    
    # Çubukların üzerine değerleri yaz
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Grafiği base64 formatına dönüştür
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def load_training_history():
    """Eğitim geçmişini yükler"""
    try:
        with open('models/training_history.json', 'r') as f:
            return json.load(f)
    except:
        return None

def create_training_plot():
    """Eğitim sürecindeki doğruluk ve hata grafiklerini oluşturur"""
    history = load_training_history()
    if not history:
        return None
    
    plt.figure(figsize=(15, 6))
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], 'b-', label='Eğitim Doğruluğu', linewidth=2)
    plt.plot(history['val_accuracy'], 'r-', label='Doğrulama Doğruluğu', linewidth=2)
    plt.title('Model Doğruluk Grafiği', fontsize=12, pad=15)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Doğruluk', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Her 5 epoch'ta bir değer göster
    epochs = range(1, len(history['accuracy']) + 1)
    for i in range(0, len(epochs), 5):
        plt.text(epochs[i], history['accuracy'][i], f'{history["accuracy"][i]:.2f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(epochs[i], history['val_accuracy'][i], f'{history["val_accuracy"][i]:.2f}', 
                ha='center', va='top', fontsize=8)
    
    # Hata grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], 'b-', label='Eğitim Hatası', linewidth=2)
    plt.plot(history['val_loss'], 'r-', label='Doğrulama Hatası', linewidth=2)
    plt.title('Model Hata Grafiği', fontsize=12, pad=15)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Hata', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Her 5 epoch'ta bir değer göster
    for i in range(0, len(epochs), 5):
        plt.text(epochs[i], history['loss'][i], f'{history["loss"][i]:.2f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(epochs[i], history['val_loss'][i], f'{history["val_loss"][i]:.2f}', 
                ha='center', va='top', fontsize=8)
    
    plt.tight_layout()
    
    # Grafiği base64 formatına dönüştür
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

def process_image(file_path):
    """Görüntüyü işle ve modeller için hazırla"""
    try:
        # Görüntüyü oku ve ön işle
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Görüntü okunamadı")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Derin öğrenme modeli için görüntü
        img_dl = cv2.resize(img, (224, 224))  # Model giriş boyutuna göre ayarla
        img_dl = img_dl / 255.0
        img_dl = np.expand_dims(img_dl, axis=0)
        
        # ML modelleri için görüntü
        img_ml = cv2.resize(img, (128, 128))
        img_ml = img_ml / 255.0
        img_ml = extract_features(np.array([img_ml]))
        
        return img_ml, img_dl
    except Exception as e:
        print(f"Görüntü işleme hatası: {str(e)}")
        raise

def get_predictions(img_ml, img_dl):
    """Tüm modellerden tahmin al"""
    try:
        # Derin öğrenme modeli tahmini
        dl_pred = deep_learning_model.predict(img_dl, verbose=0)
        dl_pred_class = np.argmax(dl_pred[0])
        dl_pred_label = label_encoder.inverse_transform([dl_pred_class])[0]
        dl_confidence = float(np.max(dl_pred[0]) * 100)
        
        # SVM modeli tahmini
        svm_pred = label_encoder.inverse_transform(svm_model.predict(img_ml))[0]
        svm_prob = np.max(svm_model.predict_proba(img_ml)) * 100
        
        # Random Forest modeli tahmini
        rf_pred = label_encoder.inverse_transform(rf_model.predict(img_ml))[0]
        rf_prob = np.max(rf_model.predict_proba(img_ml)) * 100
        
        predictions = {
            'dl': {'prediction': dl_pred_label, 'confidence': dl_confidence},
            'svm': {'prediction': svm_pred, 'confidence': float(svm_prob)},
            'rf': {'prediction': rf_pred, 'confidence': float(rf_prob)}
        }
        
        # Tahmin grafiğini oluştur
        prediction_chart = create_prediction_chart(predictions)
        predictions['prediction_chart'] = prediction_chart
        
        # Eğitim grafiklerini ekle
        training_plot = create_training_plot()
        if training_plot:
            predictions['training_plot'] = training_plot
        
        return predictions
    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        raise

@app.route('/')
def home():
    # Eğitim grafiklerini ana sayfada göster
    training_plot = create_training_plot()
    return render_template('index.html', training_plot=training_plot)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img_ml, img_dl = process_image(filepath)
            predictions = get_predictions(img_ml, img_dl)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(predictions)
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 