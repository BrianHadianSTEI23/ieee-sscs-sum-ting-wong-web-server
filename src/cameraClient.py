
from utility import Utility
from screenClient import ScreenClient
from sleepDetectorEngine import SleepDetectorEngine
from gyroClient import GyroClient

from typing import Final
import cv2
import time
import mediapipe as mp
import numpy as np

class CameraClient:

    ESP32_IP : Final[str] = "192.168.1.5"          # kamera ip addr (fallback value)
    PORT     : Final[int] = 1234

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
                    prep_time=Utility.PREP_TIME):
        key    = phase["key"]
        durasi = phase["durasi"]
        feats  = []
        cal_miss = 0

        # ── 1. Instruksi (lanjut OTOMATIS, tanpa tombol) ──
        if not ScreenClient.screen_prep(sock, phase, prep_time):
            return None

        # ── 2. Hitung mundur, diselaraskan dengan beep di ESP32 ──
        if gyro is not None:
            gyro.buzz(Utility.BUZZ_COUNT)          # beep .. beep .. BEEP-panjang
        if not ScreenClient.screen_countdown(sock, phase, ScreenClient.COUNTDOWN_LEAD):
            return None
        # Beep panjang berbunyi TEPAT sekarang = aba-aba "MULAI"

        # ── Perekaman ──
        total = Utility.SETTLE_TIME + durasi
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= total:
                break

            frame = ScreenClient.read_frame(sock)
            if frame is None:
                return None
            h, w = frame.shape

            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Deteksi dengan CLAHE + fallback IMAGE, sama seperti mode deteksi,
            # supaya fase "mata tertutup" tidak kehilangan banyak sampel.
            lms, cal_miss = CameraClient.detect_face(detector, detector_img, frame, ts, cal_miss)

            disp = bgr.copy()
            vec = None

            if lms is not None:
                vec, boxes = SleepDetectorEngine.extract_features(lms, frame, w, h)
                if boxes is not None:
                    for (x0, y0, x1, y1) in boxes:
                        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 200, 255), 1)
                for idx in SleepDetectorEngine.LEFT_EYE + SleepDetectorEngine.RIGHT_EYE:
                    lm = lms[idx]
                    cv2.circle(disp, (int(lm.x * w), int(lm.y * h)), 2,
                            (0, 255, 0), -1)

            recording = elapsed >= Utility.SETTLE_TIME
            if recording and vec is not None:
                feats.append(vec)

            ScreenClient.draw_panel(disp, 0, 80)
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
    def record_head_phase(sock, gyro, phase, prep_time=Utility.PREP_TIME):
        pitches, feats = [], []

        # Buang sampel lama supaya tidak tercampur fase sebelumnya
        gyro.drain_samples()

        # ── 1. Instruksi (lanjut OTOMATIS, tanpa tombol) ──
        st = gyro.get_state()
        ok0 = st["connected"] and st["age"] is not None and st["age"] < GyroClient.GYRO_MAX_AGE
        sub = ("gyro OK  pitch=%.0f  rate=%.0f" % (st["pitch"], st["rate"])
            if ok0 else "GYRO TIDAK TERHUBUNG - data tidak akan terekam")
        if not ScreenClient.screen_prep(sock, phase, prep_time, sub=sub):
            return None

        # ── 2. Hitung mundur, diselaraskan dengan beep di ESP32 ──
        gyro.buzz(Utility.BUZZ_COUNT)
        if not ScreenClient.screen_countdown(sock, phase, ScreenClient.COUNTDOWN_LEAD):
            return None
        # Beep panjang berbunyi TEPAT sekarang = aba-aba "MULAI"

        # ── Perekaman ──
        total = Utility.SETTLE_TIME + phase["durasi"]
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= total:
                break

            frame = ScreenClient.read_frame(sock)
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

            ScreenClient.draw_panel(disp, 0, 80)
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
        