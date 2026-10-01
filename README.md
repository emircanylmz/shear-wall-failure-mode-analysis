# Shear Wall Failure-Mode Analysis

Betonarme perde duvar deneylerini kullanarak göçme modu sınıflandırması ve açıklayıcı anomali incelemesi yapan, veri sızıntısına karşı grup ayrımlı bir makine öğrenmesi çalışmasıdır.

## Öne çıkanlar

- Aynı yazara veya aynı dokuz özellik vektörüne ait kayıtlar eğitim ve test arasında bölünmez.
- Random Forest ve XGBoost, beş katlı grup çapraz doğrulamayla değerlendirilir.
- Ölçekleme, kategorik kodlama, örnekleme ve sınıf ağırlıkları yalnızca ilgili eğitim fold'unda öğrenilir.
- MLP için birbirinden ayrık eğitim, doğrulama ve test grupları kullanılır; erken durdurma yalnızca doğrulama sonucuna dayanır.
- Isolation Forest ve LOF çıktıları doğrulanmış hata olarak değil, uzman incelemesi için aday kayıt olarak ele alınır.
- Sonuçlarla birlikte fold metrikleri, grup-bootstrap güven aralıkları, tahminler ve yeniden üretilebilirlik metadata'sı kaydedilir.

## Veri

- `data/raw/Shear_Wall_Database.xlsx`: değiştirilmemiş kaynak çalışma kitabı.
- `data/processed/Shear_Wall_Database_clean.xlsx`: benzersiz `RowID`, korunan `SourceID`, kalite bayrakları ve veri sözlüğü içeren temiz sürüm.

Model girdileri `M/Vlw`, `lw/tw`, `ρvwFy,vw/fc`, `ρhwFy,vw/fc`, `ρvcFy,vc/fc`, `ρhcFy,hc/fc`, `P/fcAg`, `Ab/Ag` ve `Section` alanlarıdır. `Author`, `Specimen`, `RowID` ve `SourceID` tahmin girdisi değildir. Ayrıntılar için [veri sözlüğüne](docs/DATA_DICTIONARY.md) bakın.

## Kurulum

Python 3.12 ile doğrulanmıştır.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Kullanım

Ana analiz, varsayılan veri ve çıktı yollarıyla:

```bash
python main.py
```

XGBoost karşılaştırması:

```bash
python xgb.py
```

Farklı dosya veya çıktı klasörü vermek için:

```bash
python main.py --data path/to/data.xlsx --output-dir path/to/results
```

Hızlı doğrulama için `--quick`, MLP çalıştırmadan ana analizi yürütmek için `--skip-mlp` kullanılabilir. Kısıtlı macOS çalışma alanlarında XGBoost/OpenMP paylaşımlı bellek izni gerekebilir.

Temiz çalışma kitabını kaynaktan yeniden üretmek için Node.js ve `@oai/artifact-tool` bulunan bir ortamda:

```bash
node scripts/build_clean_workbook.mjs
```

## Doğrulanmış sonuçlar

| Model | Değerlendirme | Accuracy | Balanced accuracy | Macro-F1 | ROC-AUC OVR macro |
|---|---|---:|---:|---:|---:|
| Random Forest | 5-fold grouped OOF | 0.4733 | 0.4596 | 0.4165 | 0.7352 |
| XGBoost | 5-fold grouped OOF | 0.4478 | 0.4183 | 0.3780 | 0.7145 |
| MLP | Ayrık grup test seti | 0.3377 | 0.3608 | 0.3577 | 0.6412 |

Random Forest macro-F1 için grup-bootstrap %95 aralığı `0.3014–0.5820`, XGBoost için `0.2605–0.5654` olarak bulundu. Geniş aralıklar; sınıf dengesizliği, küçük `Sliding` sınıfı ve çalışma/yazar grupları arasındaki değişkenlik nedeniyle tek nokta tahminlerinin ihtiyatla yorumlanması gerektiğini gösterir. Tam sonuçlar [sonuç raporunda](docs/RESULTS.md), yöntem ayrıntıları [metodoloji belgesinde](docs/METHODOLOGY.md) yer alır.

## Çıktılar

- `outputs/shear_wall_analysis/`: Random Forest, MLP ve anomali incelemesi sonuçları.
- `outputs/xgboost_analysis/`: XGBoost karşılaştırma sonuçları.
- `*.json`: yapılandırma, sürüm, veri özeti ve metrik metadata'sı.
- `*.csv`: fold metrikleri, OOF tahminleri, önem değerleri ve inceleme tabloları.
- `*.png`: sonuç görselleri.

Serileştirilmiş model dosyaları (`.joblib`, `.pth`) platform ve güvenlik nedenleriyle Git'e eklenmez; komutlar çalıştırıldığında yerel olarak yeniden oluşturulur. Güvenilmeyen kaynaklardan gelen pickle/joblib dosyalarını yüklemeyin.

## Test

```bash
python -m pytest -q
```

Testler veri şemasını, benzersiz kimlikleri, grup ayrımını, yinelenen özellik vektörlerinin birlikte tutulmasını ve fold-local pipeline'ın temel çalışmasını doğrular.

## Proje yapısı

```text
data/                 Kaynak ve temiz veri çalışma kitapları
docs/                 Yöntem, sonuç ve veri sözlüğü
outputs/              Yeniden üretilebilir tablo, metadata ve görseller
scripts/              Veri hazırlama yardımcıları
tests/                Otomatik kontroller
main.py               Random Forest, MLP ve anomali akışı
xgb.py                XGBoost karşılaştırma akışı
```

Katkı süreci için [CONTRIBUTING.md](CONTRIBUTING.md), güvenlik bildirimleri için [SECURITY.md](SECURITY.md) dosyasına bakın. Anomali bayrakları doğrulanmış veri hatası değildir ve kayıtlar bu bayraklara dayanarak sınıflandırma verisinden otomatik çıkarılmaz.
