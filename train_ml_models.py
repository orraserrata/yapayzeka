import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessing import load_data, extract_features

def train_models():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, le = load_data()
    
    # Extract features for ML models
    X_train_features = extract_features(X_train)
    X_test_features = extract_features(X_test)
    
    # Train SVM model
    print("Training SVM model...")
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train_features, y_train)
    svm_pred = svm_model.predict(X_test_features)
    print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_pred, target_names=le.classes_))
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_features, y_train)
    rf_pred = rf_model.predict(X_test_features)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=le.classes_))
    
    # Save models and label encoder
    joblib.dump(svm_model, 'models/svm_model.joblib')
    joblib.dump(rf_model, 'models/rf_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    
    return svm_model, rf_model, le

if __name__ == "__main__":
    import os
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    train_models() 