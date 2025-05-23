import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json

def attention_block(x, filters):
    """Attention mekanizması"""
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(filters, activation='relu')(attention)
    attention = layers.Dense(filters, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, filters))(attention)
    return layers.multiply([x, attention])

def residual_block(x, filters, kernel_size=3):
    """Residual blok"""
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def create_deep_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # İlk konvolüsyon katmanı
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # İkinci konvolüsyon bloğu
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Üçüncü konvolüsyon bloğu
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    
    # Düzleştirme ve yoğun katmanlar
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def train_deep_learning_model():
    # Veri dizinlerini tanımla
    data_dir = 'PlantVillage'
    
    # Görüntü boyutu ve batch size tanımla
    img_size = (128, 128)  # Tutarlı görüntü boyutu
    batch_size = 32
    
    # Veri artırma ile eğitim için veri üreteci oluştur
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.7, 1.3],
        validation_split=0.2
    )
    
    # Test için veri üreteci (sadece ölçeklendirme)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Eğitim verilerini yükle
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Doğrulama verilerini yükle
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Sınıf isimlerini kaydet
    class_indices = train_generator.class_indices
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    # Model oluştur ve derle
    model = create_deep_cnn_model(img_size + (3,), len(class_indices))
    
    # Adam optimizer'ı ve öğrenme oranını ayarla
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Callbacks tanımla
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Modeli eğit
    print("Model eğitimi başlıyor...")
    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Eğitim geçmişini kaydet
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open('models/training_history.json', 'w') as f:
        json.dump(history_dict, f)
    
    # Modeli değerlendir
    print("\nModel değerlendiriliyor...")
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f'\nTest doğruluğu: {test_acc:.4f}')
    
    # Son modeli kaydet
    model.save('models/deep_learning_model.h5')
    print("\nModel kaydedildi: models/deep_learning_model.h5")
    
    return model, history

if __name__ == "__main__":
    import os
    
    # Models dizinini oluştur
    if not os.path.exists('models'):
        os.makedirs('models')
        
    print("Eğitim başlıyor...")
    train_deep_learning_model() 
    train_deep_learning_model() 