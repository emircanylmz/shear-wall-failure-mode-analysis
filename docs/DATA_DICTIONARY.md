# Veri sözlüğü

Temiz çalışma kitabındaki `Database` sayfası aşağıdaki alanları içerir. Tam açıklamalar aynı dosyanın `Data Dictionary` sayfasında da yer alır.

| Alan | Tür | Analizdeki rol | Açıklama |
|---|---|---|---|
| `RowID` | Tamsayı | Kimlik | Temiz veri içinde 1–393 benzersiz kayıt kimliği |
| `SourceID` | Tamsayı | İzlenebilirlik | Kaynak çalışma kitabındaki kimlik; bazı değerler yinelenir |
| `SourceIDRepeated` | İkili | Kalite bayrağı | Aynı SourceID başka satırda da bulunuyorsa 1 |
| `Author` | Metin | Grup | Deneyin raporlandığı çalışma/yazar; model girdisi değildir |
| `Specimen` | Metin | Kimlik | Çalışma içindeki numune adı; model girdisi değildir |
| `FailureMode` | Kategori | Hedef | 1=Flexural, 2=Shear, 3=Flex-Shear, 4=Sliding |
| `FailureModeName` | Metin | Hedef etiketi | Hedef kodunun okunabilir karşılığı |
| `M/Vlw` | Sayısal | Model girdisi | Moment-kesme oranı |
| `lw/tw` | Sayısal | Model girdisi | Duvar uzunluğu-kalınlığı oranı |
| `ρvwFy,vw/fc` | Sayısal | Model girdisi | Düşey gövde donatısı normalize oranı |
| `ρhwFy,vw/fc` | Sayısal | Model girdisi | Yatay gövde donatısı normalize oranı |
| `ρvcFy,vc/fc` | Sayısal | Model girdisi | Düşey sınır/köşe donatısı normalize oranı |
| `ρhcFy,hc/fc` | Sayısal | Model girdisi | Yatay sınır/köşe donatısı normalize oranı |
| `P/fcAg` | Sayısal | Model girdisi | Normalize eksenel yük oranı |
| `Section` | Kategori | Model girdisi | Kesit tipi: R, B veya F |
| `Ab/Ag` | Sayısal | Model girdisi | Sınır bölgesi alan oranı |
| `FeatureVectorGroupSize` | Tamsayı | Kalite/grup alanı | Aynı dokuz model girdisini paylaşan kayıt sayısı |

Veri özeti: 393 kayıt, 75 yazar/çalışma, 12 yinelenen `SourceID` satırı ve yinelenen özellik vektörü içinde yer alan 23 satır. Model alanlarında eksik hücre ve geçersiz hedef kodu bulunmaz.
