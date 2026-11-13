# Aynı Problem, Dört Öğrenme Yolu: El Yazısı Rakamların SNN ile Sınıflandırılması

Konya Teknik Üniversitesi · Yazılım Mühendisliği (YL) — *Beyin Merkezli Yapay Zekâ: Nöro-Mimetik Sistem Tasarımı* dersi kapsamında geliştirilen bu projede, `scikit-learn` `digits` ($8\times 8$, 10 sınıf) veri kümesi üzerinde dört farklı öğrenme yaklaşımı; **doğruluk, karar gecikmesi ($T$), enerji vekili (MAC / SynOp)** ve **öğrenme kuralının yerelliği** eksenlerinde karşılaştırılmıştır[cite: 1].

## Uygulanan Yöntemler

1. **G1 · Referans MLP (Saf NumPy):** `64 → 128 → 10` mimarisinde ReLU gizli katman, doğrusal çıkış katmanı ve mini-batch geri yayılım[cite: 1].
2. **G2 · Spike Kodlama:** Poisson oran kodu ($1/\sqrt{T}$ hata ölçeklenmesi), Deterministik IF / Sigma-Delta oran kodu ($1/T$ hata ölçeklenmesi) ve İlk Spike Zamanı (TTFS) kodlayıcıları[cite: 1].
3. **G3 · ANN → SNN Dönüşümü:** Çıkarma ile sıfırlama (*reset by subtraction*) kullanan IF nöron katmanı ve eğitim kümesi 99,9. yüzdelik ($\lambda$) veri tabanlı ağırlık/bias ölçeklemesi[cite: 1].
4. **G4 · Vekil Türevle Doğrudan LIF Eğitimi:** $T=25$, $\beta=0.9$ parametreleriyle Zamanda Açılımlı Geri Yayılım (BPTT), hızlı sigmoid vekil türev genişliği ($\text{eğim} \in \{1, 10, 100\}$) analizi ve gerçek türev (`0`) ablasyon deneyi[cite: 1].
5. **Bonus · Beyincik Yaklaşımı:** Geri yayılım kullanmadan sabit rastgele LIF granül hücreleri ve yerel tırmanıcı lif delta kuralı ile öğrenme[cite: 1].

## Özet Sonuçlar (`268276001006`, `silme_orani = 0.13`)[cite: 3]

| Model / Yöntem | Karar Süresi | Standart Test Doğruluğu | Bozuk Test (`-%13` Piksel) | İşlem / Enerji Vekili | Öğrenme Kuralı |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **G1 · Referans MLP** | 1 adım | `%97.41`[cite: 3] | `%88.89`[cite: 3] | `9472 MAC` / `43.571,2 pJ`[cite: 3] | Yerel Değil (BP)[cite: 3] |
| **G3 · ANN→SNN ($T=50$)** | 50 adım[cite: 3] | `%97.41`[cite: 3] | `%89.26`[cite: 3] | `136.081 SynOp` / `122.473,3 pJ`[cite: 3] | Yerel Değil (ANN BP)[cite: 3] |
| **G3 · ANN→SNN ($T=200$)** | 200 adım[cite: 3] | `%97.59`[cite: 3] | `%89.38`[cite: 3] | `545.155 SynOp` / `490.639,1 pJ`[cite: 3] | Yerel Değil (ANN BP)[cite: 3] |
| **G4 · Vekil Türev LIF ($\text{Eğim}=10$)** | 25 adım[cite: 3] | `%98.52`[cite: 3] | `%93.21`[cite: 3] | `71.120 SynOp` / `64.007,9 pJ`[cite: 3] | Yerel Değil (BPTT)[cite: 3] |
| **Bonus · Beyincik ($H=128$)** | 25 adım[cite: 3] | `%95.37`[cite: 3] | `%86.36`[cite: 3] | `70.225 SynOp` / `63.202,4 pJ`[cite: 3] | Tam Yerel (Delta)[cite: 3] |

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
jupyter notebook 268276001006_AdSoyad_NM_Odev1.ipynb