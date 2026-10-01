# Güvenlik politikası

## Desteklenen sürüm

Güvenlik düzeltmeleri yalnızca `main` dalındaki güncel sürüme uygulanır.

## Bildirim

Bir güvenlik açığı bulursanız API anahtarı, token, kişisel veri veya istismar ayrıntılarını herkese açık issue içinde paylaşmayın. Depo için GitHub Private Vulnerability Reporting etkinse güvenlik bildirimini bu kanaldan gönderin; değilse maintainer'ın GitHub profili üzerinden özel iletişim kurun.

## Model dosyaları

Python pickle, joblib ve benzeri serileştirme biçimleri yükleme sırasında kod çalıştırabilir. Bu nedenle depo model ikililerini izlemez. Yalnızca kendinizin bu kaynak koddan ürettiği veya tamamen güvendiğiniz dosyaları yükleyin.

## Gizli bilgiler

Proje çalışmak için harici API anahtarı gerektirmez. Yerel gizli değerler `.env` benzeri, Git tarafından dışlanan dosyalarda tutulmalı ve asla commit edilmemelidir.
