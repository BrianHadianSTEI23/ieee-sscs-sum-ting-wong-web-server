
import os
import urllib
from typing import Final
import numpy as np

class Utility:

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

    MODEL_PATH : Final[str] = "face_landmarker.task"
    MODEL_URL  : Final[str] = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

    PREP_TIME : Final[float]       = 3.0     # jeda membaca instruksi antar fase 
    SETTLE_TIME : Final[float]      = 1.2


    # ═════════════════════════════════════════════
    #  KALIBRASI DIPANDU BUZZER (tanpa tombol spasi)
    #
    #  Alur satu fase:
    #    [instruksi ditampilkan]  -> PREP_TIME detik
    #    beep .. beep .. BEEP     -> COUNTDOWN (mulai rekam saat beep panjang)
    #    [merekam]                -> SETTLE_TIME + durasi fase
    #    beeeep panjang           -> fase selesai
    # ═════════════════════════════════════════════
    BUZZ_BOOT    : Final[str] = "B"    # melodi mulai kalibrasi (~1.0 s)
    BUZZ_COUNT   : Final[str] = "C"    # hitung mundur 3..2..1  (2.6 s)
    BUZZ_END     : Final[str] = "E"    # fase selesai           (0.8 s)
    BUZZ_SUCCESS : Final[str] = "S"    # kalibrasi berhasil     (~1.4 s)
    BUZZ_FAIL    : Final[str] = "F"    # kalibrasi gagal        (1.0 s)

    def __init__(self):
        pass

    # ═════════════════════════════════════════════
    #  UTILITAS DASAR
    # ═════════════════════════════════════════════
    @staticmethod
    def ensure_model():
        if not os.path.exists(Utility.MODEL_PATH):
            print("Mengunduh model face_landmarker.task (sekali saja) ...")
            urllib.request.urlretrieve(Utility.MODEL_URL, Utility.MODEL_PATH)
            print("Selesai.\n")


    @staticmethod
    def recv_all(sock, size):
        data = b''
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                return None
            data += packet
        return data


    # ═════════════════════════════════════════════
    #  Mathematical functions
    # ═════════════════════════════════════════════
    @staticmethod
    def euclidean(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    # ═════════════════════════════════════════════
    #  ANALISIS + LDA
    # ═════════════════════════════════════════════
    @staticmethod
    def separability(a, b):
        """d-prime: seberapa terpisah dua distribusi 1-D."""
        va, vb = np.var(a), np.var(b)
        denom = np.sqrt(0.5 * (va + vb)) + 1e-9
        return abs(np.mean(a) - np.mean(b)) / denom

    @staticmethod
    def fit_lda(X_awake, X_closed):
        """
        Fisher LDA: cari arah w yang paling memisahkan kelas terjaga vs tertutup.
        Fitur distandardisasi dulu agar skala tidak mendominasi.
        """
        X_all = np.vstack([X_awake, X_closed])
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0) + 1e-9

        A = (X_awake  - mu) / sd
        C = (X_closed - mu) / sd

        mA, mC = A.mean(axis=0), C.mean(axis=0)
        Sw = np.cov(A, rowvar=False) * len(A) + np.cov(C, rowvar=False) * len(C)
        Sw /= (len(A) + len(C))
        Sw += np.eye(Utility.NF) * 1e-3          # regularisasi

        w = np.linalg.pinv(Sw) @ (mA - mC)
        n = np.linalg.norm(w)
        if n < 1e-9:
            return None
        w = w / n

        # Pastikan arah positif = mata terbuka
        if (A @ w).mean() < (C @ w).mean():
            w = -w

        return {"w": w, "mu": mu, "sd": sd}

    @staticmethod
    def fit_lda_generic(X_pos, X_neg, nf):
        """
        Fisher LDA umum: cari arah w yang paling memisahkan dua kelas.
        Dipakai ulang untuk gerakan kepala (angguk vs bukan-angguk).
        """
        X_all = np.vstack([X_pos, X_neg])
        mu = X_all.mean(axis=0)
        sd = X_all.std(axis=0) + 1e-9

        A = (X_pos - mu) / sd
        B = (X_neg - mu) / sd

        mA, mB = A.mean(axis=0), B.mean(axis=0)
        Sw = np.cov(A, rowvar=False) * len(A) + np.cov(B, rowvar=False) * len(B)
        Sw /= (len(A) + len(B))
        Sw += np.eye(nf) * 1e-3

        w = np.linalg.pinv(Sw) @ (mA - mB)
        n = np.linalg.norm(w)
        if n < 1e-9:
            return None
        w = w / n
        if (A @ w).mean() < (B @ w).mean():
            w = -w
        return {"w": w, "mu": mu, "sd": sd}

    @staticmethod
    def head_score(model, X):
        return ((np.atleast_2d(X) - model["mu"]) / model["sd"]) @ model["w"]

    @staticmethod
    def head_feature_vector(s):
        """Ubah satu sampel gyro jadi vektor fitur gerakan."""
        return np.array([
            abs(s.get("prate", 0.0)),   # laju perubahan sudut kemiringan
                                        #   -> BESAR saat mengangguk
                                        #   -> ~NOL saat menoleh (kepala tetap tegak)
            abs(s.get("gx", 0.0)),
            abs(s.get("gy", 0.0)),
            abs(s.get("gz", 0.0)),
            s.get("accdev", 0.0),
        ], dtype=np.float64)

    @staticmethod
    def project(model, X):
        return ((np.atleast_2d(X) - model["mu"]) / model["sd"]) @ model["w"]

    # ═════════════════════════════════════════════
    #  STATISTIK ROBUST
    # ═════════════════════════════════════════════
    @staticmethod
    def robust_low(vals, drop=0.20):
        """Nilai terendah setelah membuang `drop` bagian terbawah."""
        s = np.sort(np.asarray(vals, dtype=float))
        if len(s) < 5:
            return float(s.min())
        return float(s[int(len(s) * drop)])

    @staticmethod
    def robust_high(vals, drop=0.20):
        """Nilai tertinggi setelah membuang `drop` bagian teratas."""
        s = np.sort(np.asarray(vals, dtype=float))
        if len(s) < 5:
            return float(s.max())
        return float(s[len(s) - 1 - int(len(s) * drop)])