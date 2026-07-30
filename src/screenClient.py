
from typing import Final
from .utility import Utility
import struct
import time
import cv2
import numpy as np


class ScreenClient:
    COUNTDOWN_LEAD : Final[float]  = 2.0     # dari 'C' dikirim s/d beep panjang berbunyi

    def __init__(self):
        pass

    # ═════════════════════════════════════════════
    #  LAYAR OTOMATIS UNTUK KALIBRASI DIPANDU BUZZER
    #  (menggantikan penungguan tombol SPASI)
    # ═════════════════════════════════════════════
    def screen_prep(self, sock, phase, seconds, gyro_state_fn=None, sub=""):
        """
        Tampilkan instruksi fase selama `seconds` detik, lalu lanjut OTOMATIS.
        Kamera tetap ditampilkan supaya user bisa membenahi posisi.
        Return False kalau user menekan ESC (batal).
        """
        t0 = time.time()
        while True:
            sisa = seconds - (time.time() - t0)
            if sisa <= 0:
                return True

            frame = self.read_frame(sock)
            if frame is None:
                return False
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = frame.shape

            self.draw_panel(disp, 0, 152)
            cv2.putText(disp, phase["title"], (12, 26), cv2.FONT_HERSHEY_DUPLEX,
                        0.60, (0, 220, 255), 1, cv2.LINE_AA)
            self.draw_text_block(disp, phase["instruksi"], 12, 52)

            self.draw_panel(disp, h - 56, h)
            cv2.putText(disp, f"Bersiap... mulai dalam {sisa:0.1f}s",
                        (12, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 120), 1, cv2.LINE_AA)
            if sub:
                cv2.putText(disp, sub, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (170, 170, 170), 1, cv2.LINE_AA)
            else:
                cv2.putText(disp, "Dengarkan buzzer  |  ESC = batal",
                            (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (170, 170, 170), 1, cv2.LINE_AA)

            # Bar mundur
            p = 1.0 - max(sisa / max(seconds, 1e-6), 0.0)
            cv2.rectangle(disp, (0, h - 6), (int(w * p), h), (0, 200, 120), -1)

            cv2.imshow("Microsleep Detector", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return False


    def screen_countdown(self, sock, phase, seconds=COUNTDOWN_LEAD):
        """
        Layar hitung mundur 3..2..1 yang SELARAS dengan beep di ESP32.
        Buzzer sudah dipicu sebelum fungsi ini dipanggil.
        Return False kalau ESC.
        """
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= seconds:
                return True

            frame = self.read_frame(sock)
            if frame is None:
                return False
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = frame.shape

            # 0-1s -> "3", 1-2s -> "2", lalu beep panjang = MULAI
            angka = 3 - int(elapsed)
            angka = max(angka, 1)

            self.draw_panel(disp, 0, 60)
            cv2.putText(disp, phase["title"], (12, 24), cv2.FONT_HERSHEY_DUPLEX,
                        0.56, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, "Bersiap...", (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(disp, str(angka), (w // 2 - 26, h // 2 + 26),
                        cv2.FONT_HERSHEY_DUPLEX, 2.4, (0, 220, 255), 4, cv2.LINE_AA)

            cv2.imshow("Microsleep Detector", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return False


    def screen_wait(self, sock, seconds, judul, baris, warna=(0, 220, 255)):
        """Layar tunggu umum (mis. jeda membenahi topi di awal)."""
        t0 = time.time()
        while True:
            sisa = seconds - (time.time() - t0)
            if sisa <= 0:
                return True
            frame = self.read_frame(sock)
            if frame is None:
                return False
            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = frame.shape

            self.draw_panel(disp, 0, 140)
            cv2.putText(disp, judul, (12, 26), cv2.FONT_HERSHEY_DUPLEX,
                        0.60, warna, 1, cv2.LINE_AA)
            self.draw_text_block(disp, baris, 12, 52)

            self.draw_panel(disp, h - 42, h)
            cv2.putText(disp, f"{sisa:0.1f} detik", (12, h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 2, cv2.LINE_AA)
            p = 1.0 - max(sisa / max(seconds, 1e-6), 0.0)
            cv2.rectangle(disp, (0, h - 6), (int(w * p), h), warna, -1)

            cv2.imshow("Microsleep Detector", disp)
            if (cv2.waitKey(1) & 0xFF) == 27:
                return False


    def read_frame(self, sock):
        size_data = Utility.recv_all(sock, 4)
        if size_data is None:
            return None
        frame_size = struct.unpack("<I", size_data)[0]
        jpg = Utility.recv_all(sock, frame_size)
        if jpg is None:
            return None
        f = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if f is None:
            return None
        return cv2.flip(f, 0)     # kamera topi terbalik

    def draw_panel(self, img, y0, y1, alpha=0.6):
        ov = img.copy()
        cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (0, 0, 0), -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


    def draw_text_block(self, img, lines, x, y, scale=0.5, color=(230, 230, 230),
                        gap=21):
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (x, y + i * gap), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

    @staticmethod
    def lm_xy(landmarks, idx, w, h):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)
