# Bitki Hastalıkları Tespit Sistemi
## Derin Öğrenme Tabanlı Görüntü İşleme Projesi

### 3. Ara Rapor: Uygulama ve Test

#### Giriş
Bu raporda, projenin uygulama aşamasında gerçekleştirilen çalışmalar, karşılaşılan sorunlar ve çözüm önerileri detaylı olarak ele alınmaktadır.

#### Uygulama Aşaması

##### 1. Veri Seti Hazırlama
- PlantVillage veri seti kullanıldı
- Toplam 54,305 görüntü
- 38 farklı bitki türü ve hastalık sınıfı
- Veri seti bölünmesi:
  - Eğitim: %70
  - Doğrulama: %15
  - Test: %15

##### 2. Model Geliştirme
- CNN tabanlı derin öğrenme modeli
- Model mimarisi:
  ```python
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
      MaxPooling2D(2, 2),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(128, (3, 3), activation='relu'),
      MaxPooling2D(2, 2),
      Flatten(),
      Dense(512, activation='relu'),
      Dropout(0.5),
      Dense(38, activation='softmax')
  ])
  ```

##### 3. Model Eğitimi
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- Epochs: 50
- Early Stopping implementasyonu

#### Karşılaşılan Sorunlar ve Çözümler

##### 1. Overfitting Sorunu
**Sorun:**
- Eğitim doğruluğu: %98.5
- Doğrulama doğruluğu: %85-90
- Yüksek varyans

**Çözüm Önerileri:**
1. Veri Artırma:
   ```python
   datagen = ImageDataGenerator(
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True,
       fill_mode='nearest'
   )
   ```

2. Dropout Oranı Artırımı:
   - 0.3'ten 0.5'e çıkarıldı
   - Regularizasyon etkisi

3. Model Mimarisi Basitleştirme:
   - Katman sayısı azaltıldı
   - Filtre sayıları optimize edildi

##### 2. Görüntü Boyutu Uyumsuzluğu
**Sorun:**
- Eğitim: 224x224
- Tahmin: 128x128
- Boyut uyumsuzluğu hatası

**Çözüm:**
```python
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img
```

##### 3. Yüksek Güven Skorları
**Sorun:**
- %100'e yakın tahmin skorları
- Overfitting göstergesi

**Çözüm Önerileri:**
1. Test Seti Genişletme:
   - Daha fazla test görüntüsü
   - Farklı koşullarda çekilmiş görüntüler

2. Model Kalibrasyonu:
   - Temperature scaling
   - Platt scaling

#### Test Sonuçları

##### 1. Model Performansı
- Doğruluk: %92.5
- F1-Score: 0.91
- Precision: 0.93
- Recall: 0.90

##### 2. Tahmin Süreleri
- Ortalama: 1.2 saniye
- Maksimum: 2.5 saniye
- Minimum: 0.8 saniye

##### 3. Kaynak Kullanımı
- GPU Bellek: 2.1 GB
- CPU Kullanımı: %45
- Disk Kullanımı: 1.8 GB

#### Öneriler ve İyileştirmeler

1. **Model İyileştirmeleri:**
   - Transfer learning uygulaması
   - Model ensemble yaklaşımı
   - Hyperparameter optimizasyonu

2. **Veri Seti İyileştirmeleri:**
   - Daha fazla veri toplama
   - Veri kalitesi kontrolü
   - Sınıf dengesizliği giderme

3. **Sistem İyileştirmeleri:**
   - Batch processing implementasyonu
   - Caching mekanizması
   - Asenkron işleme

#### Sonraki Adımlar

1. **Kısa Vadeli:**
   - Model kalibrasyonu
   - Test seti genişletme
   - Performans optimizasyonu

2. **Orta Vadeli:**
   - Web arayüzü geliştirme
   - Kullanıcı geri bildirimi sistemi
   - API geliştirme

3. **Uzun Vadeli:**
   - Mobil uygulama
   - Gerçek zamanlı analiz
   - Çoklu dil desteği 