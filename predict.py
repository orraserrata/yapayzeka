import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import json
import random

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image for prediction."""
    try:
        print(f"Görüntü yükleniyor: {image_path}")
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Görüntü yüklenirken hata oluştu {image_path}: {str(e)}")
        return None

def predict_image(model_path, image_path, class_indices):
    """Make prediction on a single image."""
    try:
        print(f"Model yükleniyor: {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        img_array = load_and_preprocess_image(image_path)
        if img_array is None:
            return None, None
        
        print("Tahmin yapılıyor...")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_name = list(class_indices.keys())[predicted_class]
        return class_name, confidence
    except Exception as e:
        print(f"Tahmin yapılırken hata oluştu: {str(e)}")
        return None, None

def get_test_images(base_dir, num_images=5):
    """Get random test images from the dataset."""
    print(f"Test görüntüleri aranıyor: {base_dir}")
    test_images = []
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                selected_image = random.choice(images)
                test_images.append(os.path.join(class_path, selected_image))
    
    print(f"Bulunan test görüntüsü sayısı: {len(test_images)}")
    return random.sample(test_images, min(num_images, len(test_images)))

def main():
    # Paths
    model_path = 'models/best_model.h5'
    base_dir = 'PlantVillage'
    
    print("Sınıf indeksleri yükleniyor...")
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        print(f"Yüklenen sınıf sayısı: {len(class_indices)}")
    except Exception as e:
        print(f"Sınıf indeksleri yüklenirken hata oluştu: {str(e)}")
        return
    
    test_images = get_test_images(base_dir)
    
    if not test_images:
        print("Test görüntüsü bulunamadı!")
        return
    
    print("\nTest görüntüleri üzerinde tahminler yapılıyor:")
    print("-" * 50)
    
    for image_path in test_images:
        print(f"\nGörüntü: {os.path.basename(image_path)}")
        print(f"Gerçek sınıf: {os.path.basename(os.path.dirname(image_path))}")
        
        class_name, confidence = predict_image(model_path, image_path, class_indices)
        if class_name and confidence:
            print(f"Tahmin edilen sınıf: {class_name}")
            print(f"Güven: {confidence:.2%}")

if __name__ == "__main__":
    main() 