import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir='PlantVillage', img_size=(128, 128)):
    images = []
    labels = []
    error_files = []
    
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Processing {class_folder}...")
        file_count = 0
        
        for img_name in os.listdir(class_path):
            if not (img_name.endswith('.jpg') or img_name.endswith('.JPG') or 
                   img_name.endswith('.jpeg') or img_name.endswith('.JPEG') or
                   img_name.endswith('.png') or img_name.endswith('.PNG')):
                continue
                
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    error_files.append(img_path)
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                
                images.append(img)
                labels.append(class_folder)
                file_count += 1
                
            except Exception as e:
                error_files.append(img_path)
                print(f"Error loading image {img_path}: {str(e)}")
                continue
        
        print(f"Successfully loaded {file_count} images from {class_folder}")
    
    if not images:
        raise ValueError("No valid images found in the dataset!")
    
    print(f"\nTotal errors: {len(error_files)}")
    if error_files:
        print("Error files:")
        for f in error_files[:10]:  # Show first 10 error files
            print(f"- {f}")
        if len(error_files) > 10:
            print(f"... and {len(error_files) - 10} more")
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize images
    images = images / 255.0
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_encoded, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset Summary:")
    print(f"Total images: {len(images)}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Classes: {le.classes_}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, le

def extract_features(images):
    # Extract color features
    color_features = np.mean(images, axis=(1, 2))
    
    # Extract texture features (using standard deviation)
    texture_features = np.std(images, axis=(1, 2))
    
    # Combine features
    features = np.concatenate([color_features, texture_features], axis=1)
    
    return features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = load_data()
    print("\nFeature extraction test:")
    features = extract_features(X_train[:5])
    print(f"Feature shape for each image: {features.shape[1]}") 