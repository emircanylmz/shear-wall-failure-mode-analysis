# Katkı rehberi

Katkılar hata düzeltmesi, yöntem iyileştirmesi, test, dokümantasyon veya yeniden üretilebilirlik geliştirmesi olarak gönderilebilir.

## Geliştirme akışı

1. Depoyu fork edin ve değişiklik için kısa isimli bir dal açın.
2. Python 3.12 ortamında `python -m pip install -r requirements.txt` komutunu çalıştırın.
3. Kod değişikliğine uygun bir test ekleyin veya mevcut testi güncelleyin.
4. `python -m pytest -q` ile testleri çalıştırın.
5. PR açıklamasında değişikliğin nedenini, doğrulama yöntemini ve performans sonucuna etkisini belirtin.

## Modelleme kuralları

- `Author`, `Specimen`, `RowID` ve `SourceID` model girdisi olarak kullanılmamalıdır.
- Aynı yazar ve aynı özellik vektörleri fold'lar arasında ayrılmamalıdır.
- Ön işleme ve dengesizlik giderme adımları yalnızca eğitim bölümü üzerinde fit edilmelidir.
- Hiperparametre veya split seçimi test etiketi performansına bakılarak yapılmamalıdır.
- Metrikler accuracy ile sınırlı bırakılmamalı; balanced accuracy, macro-F1, sınıf destekleri ve belirsizlik aralıkları birlikte raporlanmalıdır.
- Anomali skorları tek başına veri silme gerekçesi sayılmamalıdır.

## Veri ve güvenlik

Gerçek API anahtarlarını, erişim tokenlarını, parolaları, özel anahtarları, kişisel dosya yollarını veya gizli veri kümelerini commit etmeyin. Serileştirilmiş pickle/joblib model dosyaları Git'e eklenmemelidir. Veri yapısını değiştiren PR'lar kaynak, dönüşüm ve geriye dönük uyumluluk etkisini belgelemelidir.
