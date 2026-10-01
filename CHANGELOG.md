# Değişiklik günlüğü

Bu proje [Semantic Versioning](https://semver.org/) yaklaşımını izlemeyi amaçlar.

## [1.0.0] - 2026-10-01

### Eklendi

- Author ve yinelenen özellik vektörü gruplarını koruyan beş katlı değerlendirme.
- Fold-local Random Forest/SMOTENC ve XGBoost sınıf ağırlığı akışları.
- Grup-bootstrap güven aralıkları ve held-out permutation importance.
- Ayrık train/validation/test grupları kullanan MLP eğitimi.
- Isolation Forest ve LOF uzlaşmasına dayalı açıklayıcı veri kalite incelemesi.
- Temiz çalışma kitabı, veri sözlüğü, otomatik testler ve yeniden üretilebilirlik metadata'sı.
- Public depo dokümantasyonu ve CI yapılandırması.

### Düzeltildi

- Satır bazlı veri ayrımından kaynaklanan yazar ve yinelenen deney sızıntısı.
- Ön işleme ve dengesizlik giderme adımlarının değerlendirme dışına taşması.
- MLP erken durdurmada test verisinin kullanılma riski.
- Isolation Forest ve LOF karar eşiklerinin yanlış yorumlanması.
- Kimlik alanlarının model girdisi olarak kullanılma riski.
