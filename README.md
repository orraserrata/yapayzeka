# Bitki Hastalığı Tespit Sistemi

Bu proje, yapay zeka modelleri kullanarak bitki hastalıklarını tespit eden bir web uygulamasıdır. Sistem, 2 farklı makine öğrenmesi modeli (SVM ve Random Forest) ve 1 derin öğrenme modeli (CNN) kullanarak bitki hastalıklarını tespit eder.

## Özellikler

- Kullanıcı dostu web arayüzü
- Gerçek zamanlı görüntü önizleme
- 3 farklı model ile tahmin
- Her model için güven skorları
- Domates, patates ve biber hastalıklarını tespit edebilme

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. Modelleri eğitin:
```bash
python train_ml_models.py
python train_deep_learning.py
```

3. Web uygulamasını başlatın:
```bash
python app.py
```

4. Tarayıcınızda `http://localhost:5000` adresine gidin

## Kullanım

1. Web arayüzünde "Bitki Fotoğrafı Seçin" butonuna tıklayın
2. Analiz etmek istediğiniz bitki yaprağının fotoğrafını seçin
3. "Analiz Et" butonuna tıklayın
4. Her üç modelin tahminlerini ve güven skorlarını görüntüleyin

## Proje Yapısı

- `data_preprocessing.py`: Veri ön işleme fonksiyonları
- `train_ml_models.py`: Makine öğrenmesi modellerinin eğitimi
- `train_deep_learning.py`: Derin öğrenme modelinin eğitimi
- `app.py`: Flask web uygulaması
- `templates/`: HTML şablonları
- `models/`: Eğitilmiş modeller
- `uploads/`: Geçici yükleme klasörü

## Desteklenen Hastalıklar

- Domates:
  - Sağlıklı
  - Bakteriyel Leke
  - Erken Yanıklık
  - Geç Yanıklık
  - Yaprak Küfü
  - Septoria Yaprak Lekesi
  - Örümcek Akarı
  - Hedef Nokta
  - Mozaik Virüsü
  - Sarı Yaprak Kıvırcık Virüsü

- Patates:
  - Sağlıklı
  - Erken Yanıklık
  - Geç Yanıklık

- Biber:
  - Sağlıklı
  - Bakteriyel Leke 