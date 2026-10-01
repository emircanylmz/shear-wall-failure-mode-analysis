# Sonuçlar

Bu sayfa depodaki tam ayarlarla üretilen doğrulanmış sonuçları özetler. Ana makine öğrenmesi sonucu, satır bazlı rastgele ayrım değil, yazar ve yinelenen özellik vektörü gruplarını koruyan değerlendirmedir.

## Sınıflandırma

| Model | Protokol | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro-F1 | ROC-AUC OVR macro |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 5-fold grouped OOF | 0.4733 | 0.4596 | 0.4181 | 0.4596 | 0.4165 | 0.7352 |
| XGBoost | 5-fold grouped OOF | 0.4478 | 0.4183 | 0.3825 | 0.4183 | 0.3780 | 0.7145 |
| MLP | Ayrık grup test seti | 0.3377 | 0.3608 | 0.3608 | 0.3608 | 0.3577 | 0.6412 |

### Grup-bootstrap %95 güven aralıkları

| Model | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Random Forest | 0.3277–0.6503 | 0.3502–0.5975 | 0.3014–0.5820 |
| XGBoost | 0.2810–0.6375 | 0.3089–0.5657 | 0.2605–0.5654 |

Random Forest bu deneyde en yüksek OOF macro-F1 değerini verir; ancak güven aralıkları belirgin biçimde örtüşür. Bu nedenle modeller arasında kesin üstünlük iddiası desteklenmez. MLP farklı bir tek holdout protokolüyle değerlendirildiğinden OOF satırlarıyla doğrudan sıralama amacıyla karşılaştırılmamalıdır.

Random Forest sınıf bazında `Flexural` için macro olmayan F1 `0.7000`, `Shear` için `0.3839`, `Flex-Shear` için `0.2888`, `Sliding` için `0.2933` üretmiştir. En küçük sınıf olan `Sliding` yalnızca 23 kayıt içerdiği için bu sınıfın tahminleri özellikle belirsizdir.

## Veri kalitesi incelemesi

Varsayılan ayarlarda Isolation Forest 32, LOF 32 kayıt işaretlemiş; iki yöntemin uzlaşma kümesinde 17 kayıt kalmıştır. Bunlar doğrulanmış hata veya aykırı deney değildir. `outputs/shear_wall_analysis/anomaly_quality_review.csv` alan uzmanının kaynak yayın ve numune bağlamıyla inceleyebilmesi için ilgili skor ve bayrakları sunar.

## İzlenebilir çıktı dosyaları

- `outputs/shear_wall_analysis/metrics_summary.json`: birleşik konfigürasyon, sürümler ve sonuçlar.
- `outputs/shear_wall_analysis/rf_grouped_fold_metrics.csv`: Random Forest fold ayrıntıları.
- `outputs/shear_wall_analysis/rf_grouped_oof_predictions.csv`: her kaydın OOF tahmini.
- `outputs/shear_wall_analysis/rf_grouped_permutation_importance.csv`: held-out önem sonuçları.
- `outputs/xgboost_analysis/xgb_metadata.json`: XGBoost sonuç ve sınıf eşlemeleri.
- `outputs/xgboost_analysis/xgb_grouped_oof_predictions.csv`: XGBoost OOF tahminleri.

İkili model dosyaları Git'te tutulmaz. Aynı model nesneleri README'deki komutlarla yeniden üretilebilir.
