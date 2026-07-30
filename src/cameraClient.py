
from .utility import Utility
from .screenClient import ScreenClient
# from .sleepDetectorEngine import SleepDetectorEngine
from .gyroClient import GyroClient

from typing import Final
import cv2
import time
import mediapipe as mp
import numpy as np

class CameraClient:

    ESP32_IP : Final[str] = "192.168.1.5"          # kamera ip addr (fallback value)
    PORT     : Final[int] = 1234

    # Landmark EAR
    LEFT_EYE  : Final[float] = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE : Final[float] = [362, 385, 387, 263, 373, 380]

    # Sudut mata (untuk menentukan ROI yang STABIL, tidak ikut mengecil
    # saat mata tertutup — ini penting supaya ROI terbuka vs tertutup
    # menutupi area yang sama dan bisa dibandingkan)
    LEFT_CORNERS  : Final[tuple[int]]  = (33, 133)     # (luar, dalam)
    RIGHT_CORNERS : Final[tuple[int]]  = (362, 263)    # (dalam, luar)

    # GRACE PERIOD: kalau wajah hilang SEBENTAR sementara mata sedang
    # tertutup, JANGAN reset timer. Kehilangan wajah tepat saat mata
    # menutup justru gejala khas microsleep — kalau di-reset, alarm tidak
    # akan pernah bunyi. Di luar jendela ini, state tetap di-reset supaya
    # menoleh/kamera terhalang tidak memicu alarm palsu.
    FACE_LOSS_GRACE : Final[int] = 0.7        # detik

    # CLAHE: perataan kontras lokal untuk membantu MediaPipe mengunci wajah
    # saat pencahayaan tidak merata (dahi terang, mata gelap).
    # PENTING: hanya dipakai untuk INPUT DETEKSI. Ekstraksi fitur mata tetap
    # memakai frame ASLI, supaya nilai fitur (dark_ratio, contrast, lap_var)
    # tidak bergeser dan profil kalibrasi tetap konsisten.
    USE_CLAHE : Final[bool] = True

    # Setelah sekian frame gagal berturut-turut, coba deteksi ulang memakai
    # detektor mode IMAGE (deteksi penuh, bukan pelacakan). Berguna saat
    # pelacakan mode VIDEO tersangkut dan tidak bisa mengunci ulang.
    FALLBACK_AFTER_MISSES : Final[int] = 3

    _clahe : Final[cv2.CLAHE] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    def __init__(self, ip = ESP32_IP, port = PORT):
        self._ip = ip
        self._port = port
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

    def enhance_for_detection(self, gray):
        """
        Perataan kontras lokal (CLAHE) untuk membantu MediaPipe mengunci wajah.
        HANYA dipakai sebagai input deteksi — ekstraksi fitur mata tetap
        memakai frame ASLI supaya nilai fitur tidak bergeser.
        """
        if not self.USE_CLAHE:
            return gray
        return self._clahe.apply(gray)


    def detect_face(self, detector, detector_img, gray, ts, miss_count):
        """
        Deteksi wajah dengan dua lapis ketahanan:
        1. Mode VIDEO (cepat, pakai pelacakan antar frame)
        2. Kalau gagal berturut-turut -> mode IMAGE (deteksi penuh dari nol)

        Return: (landmarks | None, miss_count_baru)
        """
        enhanced = self.enhance_for_detection(gray)
        rgb = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                        cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts[0] += 33
        res = detector.detect_for_video(mp_img, ts[0])
        if res.face_landmarks:
            return res.face_landmarks[0], 0

        miss_count += 1

        # Pelacakan VIDEO gagal -> coba deteksi penuh mode IMAGE
        if detector_img is not None and miss_count >= self.FALLBACK_AFTER_MISSES:
            res2 = detector_img.detect(mp_img)
            if res2.face_landmarks:
                return res2.face_landmarks[0], 0

        return None, miss_count

    # ═════════════════════════════════════════════
    #  PEREKAMAN FASE KALIBRASI (eye)
    # ═════════════════════════════════════════════
    
    def record_phase(self, sock, detector, detector_img, ts, phase, gyro=None,
                    prep_time=Utility.PREP_TIME, web_server = None, scr_client = None, 
                    cam_client = None):
        key    = phase["key"]
        durasi = phase["durasi"]
        feats  = []
        cal_miss = 0

        if scr_client is None and cam_client is None : 
            return None

        # ── 1. Instruksi (lanjut OTOMATIS, tanpa tombol) ──
        if not scr_client.screen_prep(sock, phase, prep_time):
            return None

        # ── 2. Hitung mundur, diselaraskan dengan beep di ESP32 ──
        if gyro is not None:
            gyro.buzz(Utility.BUZZ_COUNT)          # beep .. beep .. BEEP-panjang
        if not scr_client.screen_countdown(sock, phase, scr_client.COUNTDOWN_LEAD):
            return None
        # Beep panjang berbunyi TEPAT sekarang = aba-aba "MULAI"

        # ── Perekaman ──
        total = Utility.SETTLE_TIME + durasi
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= total:
                break

            frame = scr_client.read_frame(sock)
            if frame is None:
                return None
            h, w = frame.shape

            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Deteksi dengan CLAHE + fallback IMAGE, sama seperti mode deteksi,
            # supaya fase "mata tertutup" tidak kehilangan banyak sampel.
            lms, cal_miss = cam_client.detect_face(detector, detector_img, frame, ts, cal_miss)

            disp = bgr.copy()
            vec = None

            if lms is not None:
                vec, boxes = self.extract_features(lms, frame, w, h)
                if boxes is not None:
                    for (x0, y0, x1, y1) in boxes:
                        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 200, 255), 1)
                for idx in self.LEFT_EYE + self.RIGHT_EYE:
                    lm = lms[idx]
                    cv2.circle(disp, (int(lm.x * w), int(lm.y * h)), 2,
                            (0, 255, 0), -1)

            recording = elapsed >= Utility.SETTLE_TIME
            if recording and vec is not None:
                feats.append(vec)

            scr_client.draw_panel(disp, 0, 80)
            cv2.putText(disp, phase["title"], (12, 24), cv2.FONT_HERSHEY_DUPLEX,
                        0.58, (0, 220, 255), 1, cv2.LINE_AA)
            if recording:
                cv2.putText(disp, f"MEREKAM  sisa {total - elapsed:0.1f}s  n={len(feats)}",
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (0, 80, 255), 2, cv2.LINE_AA)
                cv2.circle(disp, (w - 22, 22), 8, (0, 80, 255), -1)
            else:
                cv2.putText(disp, "bersiap (belum merekam)", (12, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1,
                            cv2.LINE_AA)
            if vec is not None:
                cv2.putText(disp, f"EAR {vec[0]:.3f}  dark {vec[2]:.2f}  spr {vec[5]:.2f}",
                            (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 120), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "wajah tidak terdeteksi", (12, 72),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1,
                            cv2.LINE_AA)

            cv2.rectangle(disp, (0, h - 6), (int(w * min(elapsed / total, 1.0)), h),
                        (0, 220, 255), -1)

            # !! PUSH UPDATES TO WEB SERVER HERE
            if web_server is not None:
                web_server.update_frame(disp)
                if gyro is not None:
                    web_server.update_gyro(gyro.get_state())

            cv2.imshow("Microsleep Detector", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return None

        # ── 4. Beep panjang: fase selesai ──
        if gyro is not None:
            gyro.buzz(Utility.BUZZ_END)

        if len(feats) < 8:
            print(f"  [!] Fase '{key}': hanya {len(feats)} sampel valid. Terlalu sedikit.")
            return None

        return np.vstack(feats)

    # ═════════════════════════════════════════════
    #  PEREKAMAN FASE KALIBRASI (GYRO / HEAD)
    # ═════════════════════════════════════════════
    def record_head_phase(self, sock, gyro, phase, prep_time=Utility.PREP_TIME, web_server = None,
                          scr_client = None, cam_client = None):
        pitches, feats = [], []

        if scr_client is None and cam_client is None:
            return None

        # Buang sampel lama supaya tidak tercampur fase sebelumnya
        gyro.drain_samples()

        # ── 1. Instruksi (lanjut OTOMATIS, tanpa tombol) ──
        st = gyro.get_state()
        ok0 = st["connected"] and st["age"] is not None and st["age"] < GyroClient.GYRO_MAX_AGE
        sub = ("gyro OK  pitch=%.0f  rate=%.0f" % (st["pitch"], st["rate"])
            if ok0 else "GYRO TIDAK TERHUBUNG - data tidak akan terekam")
        if not scr_client.screen_prep(sock, phase, prep_time, sub=sub):
            return None

        # ── 2. Hitung mundur, diselaraskan dengan beep di ESP32 ──
        gyro.buzz(Utility.BUZZ_COUNT)
        if not scr_client.screen_countdown(sock, phase, scr_client.COUNTDOWN_LEAD):
            return None
        # Beep panjang berbunyi TEPAT sekarang = aba-aba "MULAI"

        # ── Perekaman ──
        total = Utility.SETTLE_TIME + phase["durasi"]
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= total:
                break

            frame = scr_client.read_frame(sock)
            if frame is None:
                return None
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = frame.shape

            g = gyro.get_state()
            ok = g["connected"] and g["age"] is not None and g["age"] < GyroClient.GYRO_MAX_AGE

            # Ambil SEMUA sampel baru (bukan hanya yang terakhir) supaya
            # puncak angguk tidak terlewat saat data datang bergerombol
            new_samples = gyro.drain_samples()
            if elapsed >= Utility.SETTLE_TIME and ok:
                for s in new_samples:
                    pitches.append(s["pitch"])
                    feats.append(Utility.head_feature_vector(s))

            scr_client.draw_panel(disp, 0, 80)
            cv2.putText(disp, phase["title"], (12, 24), cv2.FONT_HERSHEY_DUPLEX,
                        0.56, (0, 220, 255), 1, cv2.LINE_AA)
            if elapsed >= Utility.SETTLE_TIME:
                cv2.putText(disp, f"MEREKAM  sisa {total - elapsed:0.1f}s  n={len(pitches)}",
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (0, 80, 255), 2, cv2.LINE_AA)
                cv2.circle(disp, (w - 22, 22), 8, (0, 80, 255), -1)
            else:
                cv2.putText(disp, "bersiap (belum merekam)", (12, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1,
                            cv2.LINE_AA)
            cv2.putText(disp, f"pitch {g['pitch']:6.1f}   prate {g['prate']:7.1f}",
                        (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 120) if ok else (0, 165, 255), 1, cv2.LINE_AA)

            cv2.rectangle(disp, (0, h - 6), (int(w * min(elapsed / total, 1.0)), h),
                        (0, 220, 255), -1)
            
            # !! PUSH UPDATES TO WEB SERVER HERE
            if web_server is not None:
                web_server.update_frame(disp)
                if gyro is not None:
                    web_server.update_gyro(gyro.get_state())

            cv2.imshow("Microsleep Detector", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return None

        # ── 4. Beep panjang: fase selesai ──
        gyro.buzz(Utility.BUZZ_END)

        if len(feats) < 10:
            print(f"  [!] Fase '{phase['key']}': hanya {len(feats)} sampel. Gyro bermasalah?")
            return None

        return {"kind": phase["kind"],
                "pitch": np.array(pitches),
                "X":     np.vstack(feats)}
        