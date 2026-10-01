# Metodoloji

## Amaç ve hedef

Amaç, betonarme perde duvar deneylerinin dört göçme modundan birine sınıflandırılmasıdır: `1=Flexural`, `2=Shear`, `3=Flex-Shear`, `4=Sliding`. Birincil değerlendirme ölçütü macro-F1'dir; accuracy, balanced accuracy ve one-vs-rest macro ROC-AUC tamamlayıcı olarak raporlanır.

## Veri bütünlüğü

Temiz veri 393 kayıt, 75 yazar/çalışma grubu ve dokuz model girdisi içerir. `SourceID` kaynak izlenebilirliği için korunur, `RowID` temiz dosyada benzersizdir. `Author`, `Specimen`, `RowID` ve `SourceID` modele verilmez.

Yalnızca yazara göre gruplama yeterli olmayabilir: aynı dokuz girdiyi paylaşan kayıtlar farklı isimlerle yinelenebilir. Bu nedenle yazar bağlantıları ile özdeş özellik vektörü bağlantılarının geçişli bileşenleri tek değerlendirme grubu yapılır. Bu veri setinde sonuç 75 gruptur.

## Grup çapraz doğrulama

Random Forest ve XGBoost için beş katlı `StratifiedGroupKFold` kullanılır. Aday split seed'leri yalnızca test fold'larındaki sınıf dağılımı ve boyut dengesiyle puanlanır; model tahmini veya test performansı seed seçimine girmez. Her kayıt tam bir OOF tahmini alır ve aynı grup eğitim ile testte birlikte bulunamaz.

Fold'lar grup büyüklükleri nedeniyle aynı satır sayısına sahip değildir. Sonuçlar fold ortalaması yerine bütün OOF tahminleri üzerinden ana metrik olarak hesaplanır. Belirsizlik, yazar/değerlendirme gruplarını yeniden örnekleyen grup-bootstrap ile ölçülür.

## Random Forest

Sayısal alanlar standardize edilir; `Section` kategorik indeks olarak işlenir. Ön işleme ve `SMOTENC` yalnızca her fold'un eğitim verisinde fit edilir. Test fold'u hiçbir örnekleme adımına girmez. Nihai model aynı pipeline ile tam veriye fit edilir; depoda model ikilisi değil, üretim komutu ve metadata tutulur.

Özellik önemi, her fold'un ayrılmış test kısmında permutation importance ile hesaplanır. Böylece eğitim-içi impurity öneminin bilinen yanlılıkları azaltılır.

## XGBoost

Sayısal alanlar standardize edilir, `Section` one-hot kodlanır. Sentetik örnekleme yerine her fold'un yalnızca eğitim etiketlerinden hesaplanan dengeli örnek ağırlıkları kullanılır. XGBoost iç sınıf kodları `0–3`, kaynak `FailureMode` değerleri `1–4` olduğundan eşleme metadata içinde açıkça kaydedilir.

## MLP

Gruplar önce bağımsız test bölümüne, kalan gruplar eğitim ve doğrulama bölümlerine ayrılır. Ön işleyici sadece eğitim bölümüne fit edilir. Model seçimi ve erken durdurma doğrulama macro-F1 değerine dayanır; test bölümü yalnızca nihai model seçildikten sonra bir kez değerlendirilir. Checkpoint yalnızca ağırlık `state_dict` bilgisini içerir.

MLP sonucu tek bir grup holdout'una bağlıdır ve doğrudan beş fold OOF sonucu gibi yorumlanmamalıdır.

## Anomali incelemesi

Isolation Forest ve Local Outlier Factor, standardize sayısal alanlar ile one-hot `Section` girdileri üzerinde çalışır. Her yöntemin `-1` kararı anomali adayıdır; uzlaşma bayrağı iki yöntemin de işaretlediği kayıtları gösterir. Duyarlılık tablosu contamination ve komşu sayısı değiştiğinde uzlaşma kümesinin Jaccard kararlılığını raporlar.

Bu bayraklar ground-truth veri hatası değildir. Uzman alan incelemesi olmadan kayıt silmek, düzeltmek veya sınıflandırma performansını yükseltmek için kullanılamaz.

## Tekrarlanabilirlik

Ana tohum `42` olarak sabitlenir. Çıktı metadata'sı veri SHA-256 özetini, paket sürümlerini, konfigürasyonu, split seed'ini ve sınıf eşlemelerini içerir. Sonuçları yeniden üretmek için README'deki kurulum ve çalıştırma komutları kullanılmalıdır.
