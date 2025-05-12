# 🏭 Warehouse Location Problem (WLP) - 222803021 Turkan Doga Durak

Bu proje, **Depo Yerleşim Problemi (Warehouse Location Problem - WLP)** üzerine geliştirilmiştir. Amaç, müşteri taleplerini karşılayacak şekilde depo yerleşimlerini optimize ederek toplam maliyeti minimize etmektir.

## 📌 Problem Tanımı

Verilen depo ve müşteri bilgilerine göre, hangi depoların açılacağına ve her müşterinin hangi depoya atanacağına karar verilir. Hedef, depo kurulum maliyetleri ve müşteri taşıma maliyetleri dahil olmak üzere toplam maliyeti minimize etmektir.

## 📁 Proje Yapısı
Warehouse_Location_Problem/
 datasets → Veri setleri (.txt dosyaları)
 outputs → Sonuç dosyaları (.xlsx)
 main.py → Ana çözüm algoritması
 solver_utils.py → Yardımcı algoritma fonksiyonları
 requirements.txt → Gerekli Python paketleri
 README.md → Proje açıklama dosyası

 ## ⚙️ Kullanılan Teknolojiler

- Python 3.x
- Pandas
- NumPy
- PuLP (Doğrusal Programlama)
- openpyxl (Excel çıktıları için)

## 📊 Kullanılan Yöntemler

- **Greedy algoritma** ile başlangıç çözümü üretimi
- **CBC çözücü** ile Mixed-Integer Linear Programming (MILP)
- Büyük veri setlerinde hibrit yaklaşımlar

## 🚀 Kullanım

Terminalde projenin kök dizininde şu komutu çalıştırarak çözümü başlatabilirsin:

```bash
python main.py
