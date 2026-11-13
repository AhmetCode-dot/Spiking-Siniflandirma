"""
Nöromimetik · Ödev 1 yardımcı modülü
------------------------------------
Notebook'unuzun başında:

    from nm_odev_araclari import *
    v = OdevVerisi("251229059")          # kendi öğrenci numaranız
    Xtr, ytr, Xte, yte = v.ayir()

Bu modülü DEĞİŞTİRMEYİN; teslimde aynı dosya kullanılarak kontrol edilir.
Gereken paketler: numpy, scikit-learn (veri sklearn içinden gelir, indirme yok).
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

E_MAC = 4.6   # pJ · 32-bit kayan nokta çarp-topla (Hafta 3)
E_AC = 0.9    # pJ · 32-bit kayan nokta toplama = 1 SynOp (Hafta 3)


class OdevVerisi:
    """sklearn digits (8×8, 10 sınıf, 1797 örnek) · öğrenci numarasına özgü bölme ve bozulma."""

    def __init__(self, ogrenci_no: str):
        if not str(ogrenci_no).isdigit():
            raise ValueError("Öğrenci numarası yalnızca rakamlardan oluşmalı.")
        self.no = str(ogrenci_no)
        self.tohum = int(self.no) % (2**32)
        r = np.random.default_rng(self.tohum)
        d = load_digits()
        self.X = (d.data / 16.0).astype(np.float64)      # [0, 1] aralığında piksel yoğunluğu
        self.y = d.target.astype(int)
        self.goruntuler = d.images / 16.0
        self.silme_orani = round(float(r.uniform(0.10, 0.30)), 2)   # G5 dayanıklılık testi

    def ayir(self):
        """%70 eğitim / %30 test, tabakalı. Dönüş: Xtr, ytr, Xte, yte."""
        Xtr, Xte, ytr, yte = train_test_split(
            self.X, self.y, test_size=0.30, stratify=self.y, random_state=self.tohum % 10000)
        return Xtr, ytr, Xte, yte

    def bozuk_test(self, Xte):
        """Test görüntülerinin piksellerinin bir kısmını (size özgü oran) sıfırlar."""
        r = np.random.default_rng(self.tohum + 7)
        maske = r.random(Xte.shape) >= self.silme_orani
        return Xte * maske

    def __repr__(self):
        return f"OdevVerisi(no={self.no}, silme_orani={self.silme_orani})"


# ------------------------------------------------------------------ Enerji vekili
def ann_enerji(katman_boyutlari):
    """Yoğun bir ANN için örnek başına MAC sayısı ve pJ.  ör. ann_enerji([64, 128, 10])"""
    mac = sum(a * b for a, b in zip(katman_boyutlari[:-1], katman_boyutlari[1:]))
    return mac, mac * E_MAC


def snn_enerji(spike_sayilari, katman_boyutlari):
    """SNN için SynOp ve pJ.
    spike_sayilari[l]: l. katmandan çıkan toplam spike (örnek başına ortalama),
    l = 0 giriş katmanıdır. Her spike, bir sonraki katmanın tüm nöronlarına gider."""
    synop = sum(s * n for s, n in zip(spike_sayilari, katman_boyutlari[1:]))
    return synop, synop * E_AC


# ------------------------------------------------------------------ Otomatik testler
def kodlayici_test_et(kodlayici, T=200, tekrar=50):
    """Yazdığınız kodlayici(x, T) fonksiyonunu sınar.
    Beklenen: x (N, D) ∈ [0,1] → s (T, N, D), değerleri yalnızca 0/1,
    ve oran kodlayıcılar için ortalama ateşleme ≈ x."""
    r = np.random.default_rng(1)
    x = r.random((tekrar, 64))
    s = np.asarray(kodlayici(x, T))
    if s.shape != (T,) + x.shape:
        print(f"✗ Biçim {s.shape} ≠ beklenen {(T,) + x.shape}  (zaman ilk eksen olmalı)")
        return False
    if not np.isin(s, [0, 1]).all():
        print("✗ Çıktıda 0 ve 1 dışında değer var.")
        return False
    hata = np.abs(s.mean(0) - x).mean()
    print(f"✓ Biçim ve ikili değerler doğru · ortalama |oran − x| = {hata:.3f}  "
          f"(oran kodları için T={T}'de ~0.03'ün altında olmalı; TTFS için bu sayı anlamsızdır)")
    return True


def if_test_et(if_katmani):
    """if_katmani(z_dizisi, esik=1.0) → spike dizisi.  z_dizisi (T, N), sabit girişte
    ateşleme oranı ≈ clip(z/esik, 0, 1) olmalı (Hafta 3, slayt 19)."""
    T = 1000
    z = np.linspace(-0.5, 1.5, 41)
    s = np.asarray(if_katmani(np.tile(z, (T, 1)), esik=1.0))
    oran = s.mean(0)
    beklenen = np.clip(z, 0, 1)
    hata = np.abs(oran - beklenen).max()
    ok = hata < 2.0 / T
    print(("✓" if ok else "✗") + f" IF oranı ile ReLU arasındaki en büyük fark: {hata:.4f}  (sınır: {2.0 / T})")
    return ok


__all__ = ["OdevVerisi", "ann_enerji", "snn_enerji", "kodlayici_test_et", "if_test_et", "E_MAC", "E_AC"]
