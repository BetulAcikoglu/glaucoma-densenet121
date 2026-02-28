# Glaucoma Detection with DenseNet121

Bu repository, **retinal fundus görüntüleri** kullanılarak glokom hastalığının tespiti için
geliştirilmiş, **DenseNet121** tabanlı bir derin öğrenme modelini içermektedir.

Model, PyTorch framework’ü kullanılarak eğitilmiş ve değerlendirilmiştir.

---

## 📌 Project Overview (Proje Özeti)

- **Task:** Binary classification (Glaucoma / Normal)
- **Model:** DenseNet121 (ImageNet pretrained)
- **Framework:** PyTorch
- **Training Environment:** Google Colab

Model; sınıf dengesizliğini ve veri sızıntısını (data leakage) önlemek amacıyla
**stratified train / validation / test split** yöntemi kullanılarak eğitilmiş ve test edilmiştir.

---
## 📊 Dataset
- Total images: **705**
  - Glaucoma: 396
  - Normal: 309
- Split ratio:
  - Train: 80%
  - Validation: 10%
  - Test: 10%

> Dataset images are **not included** in this repository.

## 🧪 Evaluation Results (Test Set)
- **Accuracy:** 1.00
- **Precision:** 1.00
- **Recall (Sensitivity):** 1.00
- **F1-score:** 1.00
- **ROC-AUC:** 1.00

Confusion Matrix:
[[31  0]
 [ 0 40]]

Train, validation ve test setleri arasında:
- Veri sızıntısı (data leakage)
- Aynı görüntünün birden fazla sette bulunması (duplicate images)

**tespit edilmemiştir.**

## 📁 Repository Structure
.
├── densenet121_best.pt        # Trained model weights
├── splits_densenet121.json    # Train/Val/Test split definition
├── requirements.txt           # Python dependencies
└── README.md

## ⚙️ Installation
```bash
pip install -r requirements.txt
```
## ▶️ Inference

`inference.py` dosyası kullanılarak tek bir fundus görüntüsü üzerinden
**glokom / normal** tahmini yapılabilir.

Örnek kullanım:

python inference.py --image sample.jpg --model densenet121_best.pt

## 🔽 Trained Model Weights

Eğitilmiş DenseNet121 model ağırlıkları GitHub Release sayfasında paylaşılmıştır:
👉 https://github.com/BetulAcikoglu/glaucoma-densenet121/releases/tag/v1.0

File:
- `densenet121_best.pt` (PyTorch model weights)




## ⚠️ Disclaimer
Bu proje araştırma ve eğitim amaçlıdır.
Klinik kullanım için onaylanmış bir tanı sistemi değildir.

---

Modelin klinik ortamlarda kullanılabilmesi için;

Daha büyük veri setleri

Farklı popülasyonlar

Çok merkezli klinik çalışmalar


---

## 🧪 External Test Results (New / Different-domain)

Model, eğitim/validation/test setleriyle **hiç çakışmayan** ayrı bir test kümesi
(New_Test_Data) üzerinde ayrıca değerlendirilmiştir.

- Normal: **93**
- Glokom: **51**
- Toplam: **144**
- Data leakage / duplicate: **0** (hash/MD5 kontrolü)

### Results
- **Accuracy:** 0.4167

Confusion Matrix:
[[ 9 84]
[ 0 51]]


### Interpretation
- Model, bu yeni test setinde **normal sınıfını ayırt etmekte zorlanmış**
  ve yüksek sayıda **false positive** üretmiştir.
- Bu durum, genellikle **domain shift** (farklı veri kaynağı, çekim koşulları,
  cihaz veya ön işleme farkları) ve/veya **overfitting** ile ilişkilidir.
- Model, eğitildiği veri dağılımında yüksek performans gösterirken,
  farklı dağılımdan gelen verilerde genelleme yapamamıştır.

Bu sonuçlar, modelin klinik kullanımdan önce
**daha büyük ve çok merkezli veri setleri** ile
ek doğrulamalara ihtiyaç duyduğunu göstermektedir.


ile ek doğrulama yapılması gerekmektedir.
