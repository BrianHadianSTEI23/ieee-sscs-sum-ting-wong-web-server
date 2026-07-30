

from utility import Utility
from screenClient import ScreenClient
from cameraClient import CameraClient
from gyroClient import GyroClient

import cv2
import numpy as np
import json
import os
import time
from typing import Final


class SleepDetectorEngine:

    # CRITICAL VALUES FOR MICROSLEEP DETECTION
    EMA_ALPHA           : Final[float]  = 0.45    # smoothing skor
    MICROSLEEP_DURATION : Final[float]  = 1.5     # detik mata tertutup -> alert (kamera sendirian)


    HEAD_PHASES : Final[dict] = [
        {"key": "h_neutral", "kind": "normal", "durasi": 5.0,
        "title": "1/6  KEPALA TEGAK NORMAL",
        "instruksi": ["Duduk seperti sedang menyetir.",
                    "Kepala tegak, pandangan ke depan.",
                    "Gerakan kecil wajar tidak apa-apa."]},

        {"key": "h_yaw", "kind": "normal", "durasi": 6.0,
        "title": "2/6  MENOLEH KIRI-KANAN",
        "instruksi": ["Tengok kiri, lalu kanan, berulang agak cepat.",
                    "Seperti mengecek spion / blind spot.",
                    "Kepala JANGAN menunduk, hanya menoleh."]},

        {"key": "h_lookdown", "kind": "normal", "durasi": 6.0,
        "title": "3/6  LIHAT SPEEDOMETER (SADAR)",
        "instruksi": ["Tundukkan kepala PELAN ke dashboard,",
                    "tahan sebentar, lalu angkat lagi. Ulangi.",
                    "Gerakan SADAR dan TERKENDALI.",
                    "Mata tetap TERBUKA."]},

        {"key": "h_vibration", "kind": "normal", "durasi": 5.0,
        "title": "4/6  GUNCANGAN JALAN",
        "instruksi": ["Goyangkan topi/kepala naik-turun kecil,",
                    "seperti melewati jalan bergelombang.",
                    "Kepala tetap TEGAK, hanya bergetar."]},

        {"key": "h_nod", "kind": "microsleep", "durasi": 9.0,
        "title": "5/6  SIMULASI TERKANTUK (HENTAKAN)",
        "instruksi": ["Jatuhkan kepala ke depan CEPAT,",
                    "lalu sentak balik ke atas (kaget).",
                    "Hentak oleng kanan dan kiri juga",
                    "Ulangi 5-6 kali selama perekaman."]},

        {"key": "h_slump", "kind": "microsleep", "durasi": 5.0,
        "title": "6/6  KEPALA TERKULAI DIAM",
        "instruksi": ["Biarkan kepala jatuh ke depan",
                    "lalu DIAM di bawah, rileks total.",
                    "Seperti orang benar-benar tertidur."]},
    ]

    PHASES : Final[list[dict]] = [
        {"key": "neutral", "kind": "open", "durasi": 5.0,
         "title": "1/6  PANDANGAN NETRAL",
         "instruksi": ["Duduk seperti posisi mengemudi normal.",
                       "Lihat LURUS ke depan, rileks.",
                       "Boleh berkedip biasa."]},

        {"key": "wide", "kind": "open", "durasi": 3.5,
         "title": "2/6  MATA TERBUKA LEBAR",
         "instruksi": ["Buka mata SELEBAR mungkin.",
                       "Tahan, usahakan tidak berkedip."]},

        {"key": "down", "kind": "open", "durasi": 5.0,
         "title": "3/6  PANDANGAN KE BAWAH",
         "instruksi": ["Kepala TETAP TEGAK menghadap depan.",
                       "Turunkan hanya BOLA MATA ke bawah",
                       "(seperti melihat speedometer).",
                       "PENTING: mata harus tetap TERBUKA."]},

        {"key": "squint", "kind": "open", "durasi": 4.0,
         "title": "4/6  MATA DISIPITKAN",
         "instruksi": ["Sipitkan mata seperti kena silau.",
                       "Mata TETAP terbuka, jangan tertutup penuh."]},

        {"key": "blink", "kind": "blink", "durasi": 5.0,
         "title": "5/6  KEDIPAN NORMAL",
         "instruksi": ["Lihat lurus ke depan.",
                       "Berkedip normal beberapa kali.",
                       "(fase validasi, tidak dipakai hitung ambang)"]},

        {"key": "closed", "kind": "closed", "durasi": 5.0,
         "title": "6/6  MATA TERTUTUP",
         "instruksi": ["TUTUP mata dengan RILEKS,",
                       "seperti saat mengantuk / tertidur.",
                       "Jangan dipejamkan kuat-kuat."]},
    ]


    # features of eye-level face
    FEATURE_NAMES : Final[list[str]] = [
        "ear",           # geometri: eye aspect ratio
        "open_norm",     # geometri: bukaan vertikal / jarak antar mata
        "dark_ratio",    # tampilan: proporsi piksel gelap (pupil/iris)
        "contrast",      # tampilan: kontras gelap-terang dalam ROI
        "lap_var",       # tampilan: ketajaman tekstur (iris punya detail)
        "dark_y_spread", # tampilan: sebaran vertikal piksel gelap
                        #   pupil = blob bulat  -> sebaran besar
                        #   bulu mata = garis   -> sebaran kecil
    ]

    NF : Final[int] = len(FEATURE_NAMES)

    # head features
    HEAD_FEATURES : Final[list[str]] = ["abs_prate", "abs_gx", "abs_gy", "abs_gz", "accdev"]
    NHF : Final[int] = len(HEAD_FEATURES)

    # PROFILE_PATHs
    HEAD_PROFILE_PATH   : Final[str] = "calib_profile_head.json"
    PROFILE_PATH        : Final[str] = "calib_profile_lda.json"

    # Durasi untuk JALUR B (kepala menunduk bertahan + mata tertutup)
    GYRO_FUSED_EYE_DURATION  : Final[float] = 0.8
    GYRO_FUSED_HEAD_DURATION : Final[float] = 0.8

    # JALUR C (hentakan kepala): mata tertutup sesingkat ini sudah cukup
    # asalkan terjadi hentakan dalam jendela waktu di bawah.
    JERK_EYE_DURATION   : Final[float] = 0.5
    JERK_WINDOW         : Final[float] = 1.5    # detik; hentakan dianggap relevan selama ini

    # Nilai default kalau kalibrasi kepala dilewati (dipakai apa adanya)
    DEFAULT_PITCH_THRESHOLD : Final[float] = 25.0
    # (DEFAULT_RATE_THRESHOLD dihapus - deteksi angguk kini pakai LDA)

    # Landmark EAR
    LEFT_EYE  : Final[float] = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE : Final[float] = [362, 385, 387, 263, 373, 380]

    # Sudut mata (untuk menentukan ROI yang STABIL, tidak ikut mengecil
    # saat mata tertutup — ini penting supaya ROI terbuka vs tertutup
    # menutupi area yang sama dan bisa dibandingkan)
    LEFT_CORNERS  : Final[tuple[int]]  = (33, 133)     # (luar, dalam)
    RIGHT_CORNERS : Final[tuple[int]]  = (362, 263)    # (dalam, luar)

    def __init__(self):
        pass

    # ═════════════════════════════════════════════
    #  EKSTRAKSI FITUR
    # ═════════════════════════════════════════════
    def eye_roi_box(self, landmarks, corners, w, h):
        """
        Kotak ROI mata yang STABIL — ukurannya ditentukan oleh JARAK SUDUT MATA
        (yang tidak berubah saat mata menutup), bukan oleh tinggi bukaan mata.
        Dengan begitu ROI mata terbuka & tertutup mencakup area yang sama.
        """
        p1 = ScreenClient.lm_xy(landmarks, corners[0], w, h)
        p2 = ScreenClient.lm_xy(landmarks, corners[1], w, h)
        cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
        eye_w = Utility.euclidean(p1, p2)
        if eye_w < 6:
            return None

        bw = eye_w * 1.15
        bh = eye_w * 0.72

        x0 = int(round(cx - bw / 2)); x1 = int(round(cx + bw / 2))
        y0 = int(round(cy - bh / 2)); y1 = int(round(cy + bh / 2))

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w - 1, x1); y1 = min(h - 1, y1)
        if x1 - x0 < 8 or y1 - y0 < 6:
            return None
        return (x0, y0, x1, y1)


    def appearance_features(self, gray, box):
        """
        Fitur berbasis tampilan piksel di dalam ROI mata.
        Semua dinormalisasi terhadap kecerahan kulit (persentil 90)
        sehingga tahan terhadap perubahan pencahayaan.
        """
        x0, y0, x1, y1 = box
        roi = gray[y0:y1, x0:x1]
        if roi.size < 48:
            return None

        roi = cv2.GaussianBlur(roi, (3, 3), 0).astype(np.float32)

        p90 = float(np.percentile(roi, 90))
        p05 = float(np.percentile(roi, 5))
        denom = p90 + 1.0

        norm = roi / denom                    # 1.0 = kulit terang
        contrast = 1.0 - (p05 / denom)        # besar = ada area sangat gelap

        dark_mask = norm < 0.58
        dark_ratio = float(dark_mask.mean())

        # Sebaran vertikal piksel gelap:
        #   pupil (blob bulat)      -> sebaran besar
        #   garis bulu mata (tipis) -> sebaran kecil
        ys, xs = np.nonzero(dark_mask)
        rh = roi.shape[0]
        if len(ys) >= 6 and rh > 1:
            dark_y_spread = float(np.std(ys) / rh)
        else:
            dark_y_spread = 0.0

        # Tekstur: iris & pantulan cahaya menghasilkan tepi tajam
        eq = cv2.equalizeHist(gray[y0:y1, x0:x1])
        lap_var = float(cv2.Laplacian(eq, cv2.CV_64F).var() / 1000.0)

        return dark_ratio, contrast, lap_var, dark_y_spread


    def eye_aspect_ratio(self, landmarks, eye_idx, w, h):
        pts = [ScreenClient.lm_xy(landmarks, i, w, h) for i in eye_idx]
        v1 = Utility.euclidean(pts[1], pts[5])
        v2 = Utility.euclidean(pts[2], pts[4])
        hz = Utility.euclidean(pts[0], pts[3])
        if hz == 0:
            return None
        return (v1 + v2) / (2.0 * hz)


    def extract_features(self, landmarks, gray, w, h):
        """Kembalikan vektor fitur (NF,) atau None, plus kotak ROI untuk display."""
        ear_l = self.eye_aspect_ratio(landmarks, self.LEFT_EYE,  w, h)
        ear_r = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE, w, h)
        if ear_l is None or ear_r is None:
            return None, None
        ear = (ear_l + ear_r) / 2.0

        # Bukaan vertikal dinormalisasi jarak antar-mata (skala wajah)
        iod = Utility.euclidean(ScreenClient.lm_xy(landmarks, 33, w, h), ScreenClient.lm_xy(landmarks, 263, w, h))
        if iod < 10:
            return None, None
        open_l = Utility.euclidean(ScreenClient.lm_xy(landmarks, 159, w, h), ScreenClient.lm_xy(landmarks, 145, w, h))
        open_r = Utility.euclidean(ScreenClient.lm_xy(landmarks, 386, w, h), ScreenClient.lm_xy(landmarks, 374, w, h))
        open_norm = ((open_l + open_r) / 2.0) / iod

        box_l = self.eye_roi_box(landmarks, self.LEFT_CORNERS,  w, h)
        box_r = self.eye_roi_box(landmarks, self.RIGHT_CORNERS, w, h)
        if box_l is None or box_r is None:
            return None, None

        ap_l = self.appearance_features(gray, box_l)
        ap_r = self.appearance_features(gray, box_r)
        if ap_l is None or ap_r is None:
            return None, None

        dark_ratio    = (ap_l[0] + ap_r[0]) / 2.0
        contrast      = (ap_l[1] + ap_r[1]) / 2.0
        lap_var       = (ap_l[2] + ap_r[2]) / 2.0
        dark_y_spread = (ap_l[3] + ap_r[3]) / 2.0

        vec = np.array([ear, open_norm, dark_ratio, contrast, lap_var,
                        dark_y_spread], dtype=np.float64)
        return vec, (box_l, box_r)

    # ═════════════════════════════════════════════
    #  CALIBRATION FEATURES
    # ═════════════════════════════════════════════

    def run_calibration(self, sock, detector, detector_img, ts, gyro=None,
                    first_prep=None):
        """
        6 fase kalibrasi MATA, dipandu buzzer, tanpa tombol.
        first_prep: lama jeda sebelum fase PERTAMA (dipakai untuk jeda
                    membenahi topi di awal rangkaian 12 fase).
        """
        print("\n" + "=" * 60)
        print("  FASE KALIBRASI MATA  (6 tahap)")
        print("=" * 60)

        data = {}
        for i, phase in enumerate(self.PHASES):
            prep = first_prep if (i == 0 and first_prep is not None) else Utility.PREP_TIME
            arr = CameraClient.record_phase(sock, detector, detector_img, ts, phase,
                            gyro=gyro, prep_time=prep)
            if arr is None:
                return None
            data[phase["key"]] = {"kind": phase["kind"], "X": arr}
            print(f"  {phase['title']:<26} n={len(arr):3d}   "
                f"EAR mean={arr[:, 0].mean():.3f}   "
                f"dark={arr[:, 2].mean():.3f}   spread={arr[:, 5].mean():.3f}")
        return data

    def run_head_calibration(self, sock, gyro):
        """6 fase kalibrasi KEPALA, dipandu buzzer, tanpa tombol."""
        print("\n" + "=" * 60)
        print("  KALIBRASI KEPALA (gyro + accelerometer) — 6 tahap")
        print("=" * 60)

        data = {}
        for phase in self.HEAD_PHASES:
            d = CameraClient.record_head_phase(sock, gyro, phase, prep_time=Utility.PREP_TIME)
            if d is None:
                return None
            data[phase["key"]] = d
            print(f"  {phase['title']:<32} n={len(d['pitch']):3d}  "
                f"pitch[{d['pitch'].min():5.1f}..{d['pitch'].max():5.1f}]  "
                f"prate_max={d['X'][:, 0].max():6.1f}")
        return data


    # ═════════════════════════════════════════════
    #  PROFILES FEATURES
    # ═════════════════════════════════════════════

    def save_head_profile(self, p):
        with open(self.HEAD_PROFILE_PATH, "w") as f:
            json.dump(p, f, indent=2)
        print(f"Profil kepala disimpan: {self.HEAD_PROFILE_PATH}\n")


    def load_head_profile(self):
        if not os.path.exists(self.HEAD_PROFILE_PATH):
            return None
        try:
            with open(self.HEAD_PROFILE_PATH) as f:
                p = json.load(f)
            # Profil lama (format rate_threshold) tidak kompatibel -> minta ulang
            if "pitch_threshold" in p and p.get("features") == self.HEAD_FEATURES:
                return p
            print("Profil kepala lama terdeteksi (format berbeda) -> perlu kalibrasi ulang.")
        except Exception:
            pass
        return None


    def default_head_profile(self):
        return {"pitch_threshold": self.DEFAULT_PITCH_THRESHOLD,
                "nod_threshold": None, "w": None, "mu": None, "sd": None,
                "features": self.HEAD_FEATURES,
                "calm_ceiling": 0.0, "down_floor": 0.0,
                "waktu": "default (belum dikalibrasi)"}


    def save_profile(self, p):
        with open(self.PROFILE_PATH, "w") as f:
            json.dump(p, f, indent=2)
        print(f"Profil disimpan: {self.PROFILE_PATH}\n")


    def load_profile(self):
        if not os.path.exists(self.PROFILE_PATH):
            return None
        try:
            with open(self.PROFILE_PATH) as f:
                p = json.load(f)
            if all(k in p for k in ("w", "mu", "sd", "t_close", "t_open")):
                if p.get("features") == self.FEATURE_NAMES:
                    return p
        except Exception:
            pass
        return None


    def compute_head_profile(self, data):
        """
        Dua hasil:
        pitch_threshold : memisahkan kepala TEGAK dari kepala MENUNDUK
        model + nod_threshold : skor LDA yang memisahkan ANGGUKAN dari
                                SEMUA gerakan sadar, termasuk MENOLEH.
        """
        calm_keys = ["h_neutral", "h_yaw", "h_vibration"]
        down_keys = ["h_lookdown", "h_slump"]

        # ── Ambang pitch (postur menunduk) ──
        calm_pitch = np.concatenate([data[k]["pitch"] for k in calm_keys if k in data])
        down_pitch = np.concatenate([data[k]["pitch"] for k in down_keys if k in data])

        calm_ceiling = Utility.robust_high(calm_pitch, drop=0.05)
        down_floor   = Utility.robust_low(down_pitch, drop=0.25)

        print("\n" + "-" * 60)
        print("  ANALISIS KALIBRASI KEPALA")
        print("-" * 60)
        print(f"  Pitch tertinggi saat tegak     : {calm_ceiling:6.1f} deg")
        print(f"  Pitch terendah saat menunduk   : {down_floor:6.1f} deg")

        if down_floor > calm_ceiling + 3.0:
            pitch_threshold = calm_ceiling + 0.45 * (down_floor - calm_ceiling)
            print(f"  -> Ambang MENUNDUK             : {pitch_threshold:6.1f} deg  [OK]")
        else:
            pitch_threshold = self.DEFAULT_PITCH_THRESHOLD
            print(f"  -> Tumpang tindih, pakai default: {pitch_threshold:6.1f} deg  [!]")

        # ═══════════════════════════════════════════
        #  DETEKSI ANGGUKAN — pakai LDA, bukan ambang magnitudo
        #
        #  Kelas POSITIF : h_nod (angguk terkantuk)
        #  Kelas NEGATIF : SEMUA gerakan sadar, TERMASUK MENOLEH
        #
        #  Dengan memasukkan h_yaw ke kelas negatif, LDA dipaksa mencari
        #  kombinasi fitur yang membuat menoleh TIDAK terdeteksi sebagai
        #  angguk — inilah yang tidak bisa dilakukan ambang magnitudo.
        # ═══════════════════════════════════════════
        aware_keys = ["h_neutral", "h_yaw", "h_lookdown", "h_vibration"]
        X_aware = np.vstack([data[k]["X"] for k in aware_keys if k in data])
        X_nod   = data["h_nod"]["X"] if "h_nod" in data else None

        print("\n  DAYA PISAH FITUR (angguk vs gerakan sadar):")
        for i, name in enumerate(self.HEAD_FEATURES):
            d = Utility.separability(X_nod[:, i], X_aware[:, i]) if X_nod is not None else 0.0
            bar = "#" * min(int(d * 4), 30)
            print(f"    {name:<10} d'={d:5.2f}  {bar}")

        # Cek khusus: angguk vs MENOLEH saja (kasus yang bermasalah)
        if "h_yaw" in data and X_nod is not None:
            print("\n  Khusus ANGGUK vs MENOLEH (kasus bermasalah):")
            for i, name in enumerate(self.HEAD_FEATURES):
                d = Utility.separability(X_nod[:, i], data["h_yaw"]["X"][:, i])
                flag = "  <== pembeda" if d > 1.0 else ""
                print(f"    {name:<10} d'={d:5.2f}{flag}")

        model = Utility.fit_lda_generic(X_nod, X_aware, self.NHF)
        if model is None:
            print("\n  [X] LDA gagal. Pakai ambang default.")
            return {"pitch_threshold": float(pitch_threshold),
                    "nod_threshold": None, "w": None, "mu": None, "sd": None,
                    "calm_ceiling": float(calm_ceiling),
                    "down_floor": float(down_floor),
                    "features": self.HEAD_FEATURES,
                    "waktu": time.strftime("%Y-%m-%d %H:%M:%S")}

        print("\n  Bobot LDA gerakan kepala:")
        for name, wi in zip(self.HEAD_FEATURES, model["w"]):
            print(f"    {name:<10} {wi:+.3f}")

        s_nod   = Utility.head_score(model, X_nod)
        s_aware = Utility.head_score(model, X_aware)

        # Ambang: di atas hampir semua gerakan sadar, tapi masih di bawah
        # puncak angguk. Condong ke sisi aman (sedikit di atas gerakan sadar).
        aware_ceiling = float(np.percentile(s_aware, 99))
        nod_peak      = float(np.percentile(s_nod, 85))
        gap = nod_peak - aware_ceiling

        print(f"\n  Skor puncak gerakan sadar      : {aware_ceiling:+.3f}")
        print(f"  Skor puncak angguk             : {nod_peak:+.3f}")
        print(f"  Gap pemisah                    : {gap:+.3f}")

        if gap > 0.15:
            nod_threshold = aware_ceiling + 0.45 * gap
            print(f"  -> Ambang ANGGUKAN             : {nod_threshold:+.3f}  [OK]")
        else:
            nod_threshold = aware_ceiling + max(0.25, abs(aware_ceiling) * 0.2)
            print(f"  -> Kurang terpisah, ambang aman: {nod_threshold:+.3f}  [!]")
            print("     Saat fase angguk, jatuhkan kepala LEBIH CEPAT & DALAM.")

        # ── Validasi terhadap tiap fase ──
        print("\n  VALIDASI (berapa % frame dianggap ANGGUKAN):")
        for key in ["h_neutral", "h_yaw", "h_lookdown", "h_vibration", "h_nod", "h_slump"]:
            if key not in data:
                continue
            sc = Utility.head_score(model, data[key]["X"])
            pct = 100.0 * np.mean(sc > nod_threshold)
            if key == "h_nod":
                verdict = "[OK]" if pct > 5 else "[!] angguk tidak terdeteksi"
            elif key == "h_yaw":
                verdict = "[OK]" if pct < 3 else "[!] MENOLEH masih salah deteksi"
            else:
                verdict = "[OK]" if pct < 5 else "[!] false positive"
            print(f"    {key:<12} {pct:5.1f}%   {verdict}")

        print("-" * 60 + "\n")

        return {
            "pitch_threshold": float(pitch_threshold),
            "nod_threshold":   float(nod_threshold),
            "w":  model["w"].tolist(),
            "mu": model["mu"].tolist(),
            "sd": model["sd"].tolist(),
            "features":     self.HEAD_FEATURES,
            "calm_ceiling": float(calm_ceiling),
            "down_floor":   float(down_floor),
            "aware_ceiling": float(aware_ceiling),
            "nod_peak":      float(nod_peak),
            "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def analyze(self, data, exclude_open_phases=()):
        open_keys = [k for k, v in data.items()
                    if v["kind"] == "open" and k not in exclude_open_phases]
        if not open_keys:
            return None

        X_awake  = np.vstack([data[k]["X"] for k in open_keys])
        X_closed = data["closed"]["X"]

        print("\n" + "-" * 60)
        print("  DAYA PISAH TIAP FITUR  (d-prime, makin besar makin bagus)")
        print("-" * 60)
        for i, name in enumerate(self.FEATURE_NAMES):
            d = Utility.separability(X_awake[:, i], X_closed[:, i])
            bar = "#" * min(int(d * 4), 34)
            print(f"  {name:<14} d'={d:5.2f}  {bar}")

        # Fitur mana yang berhasil memisahkan MENUNDUK dari TERTUTUP?
        if "down" in data and "down" not in exclude_open_phases:
            print("\n  Khusus MENUNDUK vs TERTUTUP (kasus tersulit):")
            for i, name in enumerate(self.FEATURE_NAMES):
                d = Utility.separability(data["down"]["X"][:, i], X_closed[:, i])
                flag = "  <== penyelamat" if d > 1.2 else ""
                print(f"    {name:<14} d'={d:5.2f}{flag}")

        model = Utility.fit_lda(X_awake, X_closed)
        if model is None:
            return None

        print("\n  Bobot LDA hasil kalibrasi:")
        for name, wi in zip(self.FEATURE_NAMES, model["w"]):
            print(f"    {name:<14} {wi:+.3f}")

        # ── Ambang dari skor terproyeksi ──
        floors = {k: Utility.robust_low(Utility.project(model, data[k]["X"])) for k in open_keys}
        awake_floor = min(floors.values())
        critical    = min(floors, key=floors.get)
        closed_ceil = Utility.robust_high(Utility.project(model, X_closed))
        gap = awake_floor - closed_ceil

        print("\n" + "-" * 60)
        print("  HASIL PEMISAHAN (skor LDA)")
        print("-" * 60)
        for k, v in floors.items():
            mark = "  <-- paling kritis" if k == critical else ""
            print(f"  Lantai skor terbuka [{k:<8}] : {v:+.3f}{mark}")
        print(f"  Plafon skor tertutup          : {closed_ceil:+.3f}")
        print(f"  Gap pemisah                   : {gap:+.3f}")

        if gap <= 0.05:
            print(f"\n  [X] GAGAL: kelas masih tumpang tindih. Biang keroknya: '{critical}'.")
            return {"failed": True, "critical": critical, "gap": gap}

        t_close = closed_ceil + 0.40 * gap
        t_open  = closed_ceil + 0.62 * gap

        blink = Utility.project(model, data["blink"]["X"])
        dips  = int(np.sum(blink < t_close))
        ratio = dips / len(blink)

        print(f"\n  Ambang TUTUP (t_close)        : {t_close:+.3f}")
        print(f"  Ambang BUKA  (t_open)         : {t_open:+.3f}")
        print(f"  Kedipan menembus ambang       : {dips}/{len(blink)} ({ratio*100:.0f}%)")

        if ratio > 0.55:
            print("  [!] Kedipan terlalu sering di bawah ambang — mungkin ambang kelewat tinggi.")
        elif dips == 0:
            print("  [OK] Kedipan cepat tidak menembus ambang. Normal, tidak masalah.")
        else:
            print("  [OK] Kedipan terdeteksi wajar. Ambang sehat.")

        if gap < 0.35:
            print("  [!] Gap agak sempit. Deteksi bisa sensitif terhadap noise.")

        print("-" * 60 + "\n")

        return {
            "failed":   False,
            "w":        model["w"].tolist(),
            "mu":       model["mu"].tolist(),
            "sd":       model["sd"].tolist(),
            "t_close":  float(t_close),
            "t_open":   float(t_open),
            "gap":      float(gap),
            "awake_floor":    float(awake_floor),
            "closed_ceiling": float(closed_ceil),
            "critical_phase": critical,
            "excluded":       list(exclude_open_phases),
            "features": self.FEATURE_NAMES,
            "waktu":    time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def build_profile(self, data):
        """Coba pakai semua fase. Kalau gagal, tawarkan mengeluarkan fase biang kerok."""
        prof = self.analyze(data)
        if prof is None:
            return None
        if not prof.get("failed"):
            return prof

        culprit = prof["critical"]
        print("\n" + "!" * 60)
        print(f"  Fase '{culprit}' tidak bisa dipisahkan dari kondisi mata tertutup")
        print("  pada posisi kamera ini — secara fisik kedua kondisi memang")
        print("  terlihat nyaris identik dari sudut pandang kamera.")
        print("\n  Dua pilihan:")
        print(f"    [1] Keluarkan fase '{culprit}' dari kalibrasi.")
        print(f"        Konsekuensi: bertahan di posisi '{culprit}' lebih dari")
        print(f"        {self.MICROSLEEP_DURATION:.1f} detik akan ikut memicu alarm.")
        print("    [2] Batalkan, perbaiki posisi kamera, lalu kalibrasi ulang.")
        print("        (miringkan kamera agar mata terlihat lebih dari depan)")
        print("!" * 60)

        ans = input(f"\nKeluarkan fase '{culprit}'? [y/N] ").strip().lower()
        if ans != "y":
            return None

        prof2 = self.analyze(data, exclude_open_phases=(culprit,))
        if prof2 is None or prof2.get("failed"):
            print("\n  Masih gagal juga. Posisi kamera perlu diperbaiki.")
            return None

        print(f"\n  [OK] Kalibrasi berhasil tanpa fase '{culprit}'.")
        print(f"  INGAT: kondisi '{culprit}' sekarang dianggap mata tertutup.\n")
        return prof2

    # ═════════════════════════════════════════════
    #  MODE DETEKSI
    # ═════════════════════════════════════════════
    def run_detection(self, sock, detector, detector_img, ts, prof, gyro, head_prof):
        w_vec = np.array(prof["w"])
        mu    = np.array(prof["mu"])
        sd    = np.array(prof["sd"])
        t_close, t_open = prof["t_close"], prof["t_open"]

        PITCH_TH = head_prof["pitch_threshold"]
        NOD_TH   = head_prof.get("nod_threshold")
        nod_model = None
        if head_prof.get("w") is not None and NOD_TH is not None:
            nod_model = {"w":  np.array(head_prof["w"]),
                        "mu": np.array(head_prof["mu"]),
                        "sd": np.array(head_prof["sd"])}

        score_ema        = None
        eyes_closed      = False
        closed_start     = None
        closed_duration  = 0.0
        alert_active     = False
        alert_reason     = ""
        blink_count      = 0
        show_roi         = True

        # Pelacakan hilangnya wajah (lensa wide sering lepas kunci)
        last_face_time   = 0.0     # kapan terakhir wajah terdeteksi
        face_miss        = 0       # frame gagal berturut-turut
        face_lost_for    = 999.0
        in_grace         = False

        # State gyro
        head_down          = False
        head_down_start    = None
        head_down_duration = 0.0
        last_jerk_time     = 0.0     # kapan terakhir hentakan terdeteksi
        jerk_count         = 0
        nod_score_now      = 0.0     # skor angguk terkini (untuk tampilan)

        fps_t, fps_n, fps = time.time(), 0, 0.0

        print("MODE DETEKSI aktif.")
        print(f"  t_close={t_close:+.3f}   t_open={t_open:+.3f}")
        print(f"  pitch_th={PITCH_TH:.1f}deg", end="")
        if nod_model is not None:
            print(f"   nod_th={NOD_TH:+.3f} (LDA aktif)")
        else:
            print("   deteksi angguk NONAKTIF (belum kalibrasi kepala)")
        if prof.get("excluded"):
            print(f"  Catatan: fase {prof['excluded']} dikeluarkan dari kalibrasi.")
        print("  q=keluar  r=kalibrasi ulang  d=debug  e=toggle ROI\n")

        while True:
            frame = ScreenClient.read_frame(sock)
            if frame is None:
                return "disconnect"
            h, w = frame.shape

            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Deteksi wajah dengan CLAHE + fallback mode IMAGE
            lms, face_miss = CameraClient.detect_face(detector, detector_img, frame, ts, face_miss)

            disp = bgr.copy()
            score = None
            vec = None
            now = time.time()

            if lms is not None:
                last_face_time = now
                vec, boxes = self.extract_features(lms, frame, w, h)
                if vec is not None:
                    score = float(((vec - mu) / sd) @ w_vec)
                    if show_roi and boxes is not None:
                        for (x0, y0, x1, y1) in boxes:
                            cv2.rectangle(disp, (x0, y0), (x1, y1),
                                        (0, 200, 255), 1)
                    for idx in self.LEFT_EYE + self.RIGHT_EYE:
                        lm = lms[idx]
                        cv2.circle(disp, (int(lm.x * w), int(lm.y * h)), 2,
                                (0, 255, 0), -1)

            if score is not None:
                score_ema = score if score_ema is None else \
                            self.EMA_ALPHA * score + (1 - self.EMA_ALPHA) * score_ema
            else:
                score_ema = None

            # Sudah berapa lama wajah hilang
            face_lost_for = (now - last_face_time) if last_face_time else 999.0
            in_grace = (face_lost_for < CameraClient.FACE_LOSS_GRACE)

            # ── State machine MATA (hysteresis) — TIDAK lagi memutuskan alert
            #    sendiri, hanya melacak eyes_closed & closed_duration.
            #    Keputusan alert dipindah ke blok FUSI di bawah. ──
            if score_ema is not None:
                if not eyes_closed and score_ema < t_close:
                    eyes_closed = True
                    closed_start = now
                elif eyes_closed and score_ema > t_open:
                    if 0.04 < closed_duration < self.MICROSLEEP_DURATION:
                        blink_count += 1
                    eyes_closed, closed_start = False, None
                    closed_duration = 0.0

                if eyes_closed and closed_start is not None:
                    closed_duration = now - closed_start

            elif eyes_closed and in_grace:
                # ── GRACE PERIOD ──
                # Wajah hilang SEBENTAR sementara mata sedang tertutup.
                # Jangan reset timer: kehilangan wajah tepat saat mata menutup
                # justru gejala khas microsleep (lensa wide + mata sipit membuat
                # MediaPipe mudah lepas kunci). Timer tetap berjalan.
                if closed_start is not None:
                    closed_duration = now - closed_start

            else:
                # Wajah hilang terlalu lama, atau hilang saat mata terbuka
                # -> reset penuh supaya menoleh / kamera terhalang tidak
                #    memicu alarm palsu.
                eyes_closed, closed_start = False, None
                closed_duration = 0.0

            # ── State GYRO: lacak durasi kepala menunduk ──
            g = gyro.get_state()
            gyro_ok = g["connected"] and g["age"] is not None and g["age"] < GyroClient.GYRO_MAX_AGE

            if gyro_ok:
                # Postur menunduk (bertahan)
                if g["pitch"] > PITCH_TH:
                    if not head_down:
                        
                        head_down = True
                        head_down_start = now
                    head_down_duration = now - head_down_start
                else:
                    head_down, head_down_start = False, None
                    head_down_duration = 0.0

                # ── Deteksi ANGGUKAN pakai skor LDA ──
                # Diperiksa untuk SETIAP sampel yang masuk (bukan hanya yang
                # terakhir), supaya puncak angguk tidak terlewat. LDA sudah
                # dilatih agar MENOLEH tidak ikut terdeteksi.
                if nod_model is not None:
                    new_samples = gyro.drain_samples()
                    if new_samples:
                        X = np.vstack([Utility.head_feature_vector(s) for s in new_samples])
                        scores = Utility.head_score(nod_model, X)
                        nod_score_now = float(scores.max())
                        if nod_score_now > NOD_TH:
                            if now - last_jerk_time > 0.4:
                                jerk_count += 1
                            last_jerk_time = now
            else:
                # Gyro terputus/basi -> jangan andalkan, reset supaya tidak
                # nyangkut di state lama begitu koneksi pulih
                head_down, head_down_start = False, None
                head_down_duration = 0.0

            recent_jerk = gyro_ok and (now - last_jerk_time) < self.JERK_WINDOW

            # ── FUSI KEPUTUSAN: 3 jalur, mana pun yang lebih dulu terpenuhi ──
            # A. Kamera sendirian, threshold penuh 1.5s
            #    -> fallback aman kalau gyro putus.
            # B. Kamera + kepala MENUNDUK bertahan, 0.8s
            #    -> pola tertidur perlahan, kepala terkulai.
            # C. Kamera + HENTAKAN kepala, 0.5s
            #    -> pola microsleep klasik: kepala jatuh lalu tersentak.
            #       Paling cepat karena hentakan adalah sinyal paling khas.
            path_a = eyes_closed and closed_duration >= self.MICROSLEEP_DURATION
            path_b = (eyes_closed and closed_duration >= self.GYRO_FUSED_EYE_DURATION and
                    gyro_ok and head_down and
                    head_down_duration >= self.GYRO_FUSED_HEAD_DURATION)
            path_c = (eyes_closed and closed_duration >= self.JERK_EYE_DURATION and
                    recent_jerk)

            should_alert = path_a or path_b or path_c

            if should_alert and not alert_active:
                alert_active = True
                if path_c:
                    alert_reason = "mata+hentakan"
                elif path_b:
                    alert_reason = "mata+menunduk"
                else:
                    alert_reason = "kamera"
                print(f"[ALERT] MICROSLEEP! ({alert_reason})  "
                    f"mata={closed_duration:.2f}s  menunduk={head_down_duration:.2f}s  "
                    f"nod={nod_score_now:+.2f}")
            elif not should_alert and alert_active:
                alert_active = False
                alert_reason = ""

            gyro.set_buzzer(alert_active)

            # ── Overlay ──
            ScreenClient.draw_panel(disp, 0, 78, alpha=0.55)
            if score_ema is None:
                if eyes_closed and in_grace:
                    # Wajah hilang tapi timer TETAP jalan — beri tahu user
                    cv2.putText(disp, f"TERTUTUP  {closed_duration:.1f}s",
                                (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.62,
                                (0, 80, 255), 1, cv2.LINE_AA)
                    cv2.putText(disp, f"wajah hilang {face_lost_for:.1f}s "
                                    f"(timer lanjut)", (12, 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
                                cv2.LINE_AA)
                else:
                    cv2.putText(disp, "Wajah tidak terdeteksi", (12, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 2,
                                cv2.LINE_AA)
            else:
                col = (0, 80, 255) if eyes_closed else (0, 255, 120)
                cv2.putText(disp, f"skor {score_ema:+.2f}", (12, 26),
                            cv2.FONT_HERSHEY_DUPLEX, 0.62, col, 1, cv2.LINE_AA)
                st = f"TERTUTUP  {closed_duration:.1f}s" if eyes_closed else "TERBUKA"
                cv2.putText(disp, st, (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, col, 2, cv2.LINE_AA)
                cv2.putText(disp, f"EAR {vec[0]:.3f}   kedip {blink_count}",
                            (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (170, 170, 170), 1, cv2.LINE_AA)

            # ── Overlay status gyro ──
            if gyro_ok:
                gyro_col = (0, 80, 255) if (head_down or recent_jerk) else (0, 255, 120)
                posture = f"MENUNDUK {head_down_duration:.1f}s" if head_down else "tegak"
                nod_txt = f"  nod {nod_score_now:+.2f}" if nod_model is not None else ""
                gyro_txt = (f"gyro: {g['pitch']:.0f}deg {posture}{nod_txt}" +
                            ("  [ANGGUK]" if recent_jerk else ""))
            else:
                gyro_col = (0, 165, 255)
                gyro_txt = "gyro: TERPUTUS - cek GYRO_IP & WiFi (fallback kamera saja)"
            cv2.putText(disp, gyro_txt, (12, h - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, gyro_col, 1, cv2.LINE_AA)

            # Bar skor
            bx, by, bw_, bh_ = w - 152, 14, 132, 10
            lo = prof["closed_ceiling"] - 0.6 * prof["gap"]
            hi = prof["awake_floor"] + 0.9 * prof["gap"]
            rng = max(hi - lo, 1e-6)
            cv2.rectangle(disp, (bx, by), (bx + bw_, by + bh_), (70, 70, 70), -1)
            if score_ema is not None:
                f = int(bw_ * float(np.clip((score_ema - lo) / rng, 0, 1)))
                cv2.rectangle(disp, (bx, by), (bx + f, by + bh_),
                            (0, 80, 255) if eyes_closed else (0, 255, 120), -1)
            for tv, tc in ((t_close, (0, 200, 255)), (t_open, (255, 200, 0))):
                tx = bx + int(bw_ * float(np.clip((tv - lo) / rng, 0, 1)))
                cv2.line(disp, (tx, by - 3), (tx, by + bh_ + 3), tc, 1)
            cv2.rectangle(disp, (bx, by), (bx + bw_, by + bh_), (140, 140, 140), 1)

            if eyes_closed:
                # Target = jalur tercepat yang saat ini memenuhi syarat gyro
                if recent_jerk:
                    target = self.JERK_EYE_DURATION
                elif gyro_ok and head_down:
                    target = self.GYRO_FUSED_EYE_DURATION
                else:
                    target = self.MICROSLEEP_DURATION
                p = min(closed_duration / target, 1.0)
                cv2.rectangle(disp, (0, h - 7), (int(w * p), h), (0, 80, 255), -1)

            if alert_active:
                ov = disp.copy()
                cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 200), -1)
                cv2.addWeighted(ov, 0.30, disp, 0.70, 0, disp)
                cv2.putText(disp, "! MICROSLEEP !", (w // 2 - 148, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.15, (0, 0, 255), 3,
                            cv2.LINE_AA)
                cv2.putText(disp, f"{closed_duration:.1f}s  ({alert_reason})",
                            (w // 2 - 90, h // 2 + 38), cv2.FONT_HERSHEY_SIMPLEX,
                            0.62, (0, 120, 255), 2, cv2.LINE_AA)

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
            cv2.putText(disp, f"{fps:.0f} fps", (w - 60, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1,
                        cv2.LINE_AA)

            cv2.imshow("Microsleep Detector", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                return "quit"
            if k == ord('r'):
                return "recalibrate"
            if k == ord('e'):
                show_roi = not show_roi
            if k == ord('d') and vec is not None:
                print("[debug] " + "  ".join(
                    f"{n}={v:.3f}" for n, v in zip(self.FEATURE_NAMES, vec)) +
                    f"  skor={score_ema:+.3f}  |  gyro_ok={gyro_ok} "
                    f"pitch={g['pitch']:.1f} head_down={head_down} "
                    f"head_dur={head_down_duration:.2f}")
