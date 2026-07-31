# """
# Microsleep Detector MULTI-FITUR + LDA + FUSI GYRO
# ==================================================
# VERSI SATU MIKROKONTROLER (ESP32-S3 WROOM CAM N16R8)

# Semua sensor kini berada di SATU board:
#   - OV2640      -> streaming JPEG di port 1234
#   - GY-521      -> streaming JSON di port 1235
#   - Buzzer      -> dikendalikan laptop lewat port 1235

# Dua sumber data yang difusi:
#   1. KAMERA -> skor LDA dari EAR + fitur tampilan mata
#   2. GYRO   -> sudut menunduk (pitch) + angguk terkantuk (LDA gerakan)

#   Aturan fusi (3 jalur, mana pun yang lebih dulu terpenuhi):
#     A. Kamera SENDIRIAN, mata tertutup >= MICROSLEEP_DURATION (1.5s)
#        -> fallback aman kalau data gyro terputus.
#     B. Kamera + kepala MENUNDUK bertahan >= 0.8s
#        -> pola tertidur perlahan, kepala terkulai.
#     C. Kamera + ANGGUKAN terdeteksi, mata tertutup >= 0.5s
#        -> pola microsleep klasik: kepala jatuh lalu tersentak.

#   Buzzer dikendalikan sepenuhnya dari laptop lewat gyro.set_buzzer().

# Install:
#     pip install mediapipe opencv-python numpy

# TIDAK butuh file pendukung lain — GyroClient sudah disatukan ke file ini.
# Model face_landmarker.task diunduh otomatis saat pertama dijalankan.

# Jalankan:
#     python microsleep_single_esp32s3.py

# Kontrol saat DETEKSI:
#     q  -> keluar        r  -> kalibrasi ulang
#     d  -> debug fitur   e  -> tampil/sembunyikan kotak ROI mata
# """

# import os
# # Redam log telemetri MediaPipe (pesan "clearcut" yang mengganggu)
# os.environ["GLOG_minloglevel"] = "2"
# os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

# import socket
# # import struct
# import time
# # import json
# # import urllib.request
# import cv2
# # import numpy as np
# # import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# from src.gyroClient import GyroClient
# from src.cameraClient import CameraClient
# from src.screenClient import ScreenClient
# from src.server import Server
# from src.sleepDetectorEngine import SleepDetectorEngine
# from src.utility import Utility

# # ═════════════════════════════════════════════════════════════════
# #  GyroClient — sebelumnya file terpisah (gyro_client.py),
# #  sekarang disatukan ke sini karena hanya ada 1 mikrokontroler.
# #  Isi class TIDAK diubah sama sekali dari versi yang sudah teruji.
# # ═════════════════════════════════════════════════════════════════


# class GyroClient:
#     def __init__(self, ip, port, reconnect_interval=2.0, buffer_size=400):
#         self.ip = ip
#         self.port = port
#         self.reconnect_interval = reconnect_interval

#         self._sock = None
#         self._lock = threading.Lock()
#         self._running = False
#         self._thread = None

#         self._latest = {
#             "pitch": 0.0, "roll": 0.0, "rate": 0.0, "accdev": 0.0,
#             "gx": 0.0, "gy": 0.0, "gz": 0.0, "prate": 0.0,
#         }

#         # Antrian sampel: tidak ada yang hilang walau data datang bergerombol
#         self._samples = deque(maxlen=buffer_size)

#         self._connected = False
#         self._last_update = 0.0
#         self._buzzer_state = None

#         self._worst_gap = 0.0
#         self._msg_count = 0

#     # ─────────────────────────────────────────
#     def start(self):
#         self._running = True
#         self._thread = threading.Thread(target=self._run, daemon=True)
#         self._thread.start()

#     def stop(self):
#         self._running = False
#         with self._lock:
#             if self._sock:
#                 try:
#                     self._sock.close()
#                 except Exception:
#                     pass
#         if self._thread:
#             self._thread.join(timeout=2)

#     def _run(self):
#         attempt = 0
#         while self._running:
#             try:
#                 self._connect_and_listen()
#             except socket.timeout:
#                 attempt += 1
#                 print(f"[gyro] GAGAL #{attempt}: timeout menghubungi {self.ip}:{self.port}")
#                 print("       -> IP kemungkinan salah, atau ESP tidak di WiFi yang sama.")
#             except ConnectionRefusedError:
#                 attempt += 1
#                 print(f"[gyro] GAGAL #{attempt}: koneksi ditolak {self.ip}:{self.port}")
#                 print("       -> ESP hidup tapi port salah, atau server belum siap.")
#             except OSError as e:
#                 attempt += 1
#                 print(f"[gyro] GAGAL #{attempt}: {e}")
#                 print("       -> Cek laptop & ESP di jaringan/hotspot yang sama.")
#             except Exception as e:
#                 attempt += 1
#                 print(f"[gyro] koneksi terputus: {e}")

#             with self._lock:
#                 self._connected = False
#             if self._running:
#                 time.sleep(self.reconnect_interval)

#     def _connect_and_listen(self):
#         print(f"[gyro] Menghubungkan ke {self.ip}:{self.port} ...")
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.settimeout(5.0)
#         s.connect((self.ip, self.port))

#         s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
#         try:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
#         except OSError:
#             pass
#         try:
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
#         except OSError:
#             pass

#         s.settimeout(1.0)
#         with self._lock:
#             self._sock = s
#             self._connected = True
#         print("[gyro] Terhubung.")

#         if self._buzzer_state is not None:
#             self._send_raw(self._buzzer_state)

#         buf = b""
#         while self._running:
#             try:
#                 chunk = s.recv(4096)
#             except socket.timeout:
#                 continue
#             if not chunk:
#                 raise ConnectionError("ESP32 gyro menutup koneksi")

#             buf += chunk
#             while b"\n" in buf:
#                 line, buf = buf.split(b"\n", 1)
#                 self._parse_line(line)

#     def _parse_line(self, line):
#         try:
#             obj = json.loads(line.decode("utf-8").strip())
#         except (json.JSONDecodeError, UnicodeDecodeError):
#             return

#         try:
#             sample = {
#                 "pitch":  float(obj.get("pitch",  0.0)),
#                 "roll":   float(obj.get("roll",   0.0)),
#                 "rate":   float(obj.get("rate",   0.0)),
#                 "accdev": float(obj.get("accdev", 0.0)),
#                 "gx":     float(obj.get("gx",     0.0)),
#                 "gy":     float(obj.get("gy",     0.0)),
#                 "gz":     float(obj.get("gz",     0.0)),
#                 "prate":  float(obj.get("prate",  0.0)),
#             }
#         except (ValueError, TypeError):
#             return

#         with self._lock:
#             now = time.time()
#             if self._last_update:
#                 gap = now - self._last_update
#                 if gap > self._worst_gap:
#                     self._worst_gap = gap
#             self._latest = sample
#             self._samples.append(sample)
#             self._last_update = now
#             self._msg_count += 1

#     # ─────────────────────────────────────────
#     def drain_samples(self):
#         """Ambil SEMUA sampel sejak panggilan terakhir, lalu kosongkan."""
#         with self._lock:
#             out = list(self._samples)
#             self._samples.clear()
#             return out

#     def get_state(self):
#         """Nilai terakhir + status koneksi. Untuk tampilan/overlay."""
#         with self._lock:
#             age = time.time() - self._last_update if self._last_update else None
#             st = dict(self._latest)
#             st.update({
#                 "connected": self._connected,
#                 "age": age,
#                 "worst_gap": self._worst_gap,
#                 "msg_count": self._msg_count,
#             })
#             return st

#     def reset_diagnostics(self):
#         with self._lock:
#             self._worst_gap = 0.0
#             self._msg_count = 0

#     def buzz(self, cmd: str):
#         """
#         Kirim perintah POLA buzzer ke ESP32.
#         Pola dimainkan di ESP32 (timing presisi), laptop cukup memicu.
#           'B' melodi mulai kalibrasi   'C' hitung mundur 3..2..1
#           'E' fase selesai             'S' sukses     'F' gagal
#         """
#         self._send_raw(cmd)

#     def set_buzzer(self, on: bool):
#         val = "1" if on else "0"
#         if self._buzzer_state == val:
#             return
#         self._buzzer_state = val
#         self._send_raw(val)

#     def _send_raw(self, val):
#         with self._lock:
#             sock = self._sock
#         if sock:
#             try:
#                 sock.sendall(val.encode("ascii"))
#             except Exception as e:
#                 print(f"[gyro] gagal kirim perintah buzzer: {e}")


# # ═════════════════════════════════════════════════════════════════
# #  (akhir GyroClient)
# # ═════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════
#  KONFIGURASI — KAMERA
# ═════════════════════════════════════════════
# ─────────────────────────────────────────────
#  SATU ESP32-S3 untuk kamera + gyro + buzzer.
#  Cukup isi SATU alamat IP di sini; port kamera dan port gyro
#  dilayani oleh perangkat yang sama.
#  Lihat Serial Monitor saat boot untuk mendapatkan IP-nya.
# ─────────────────────────────────────────────
# DEVICE_IP = "192.168.1.5"

# ESP32_IP = DEVICE_IP          # kamera
# PORT     = 1234

# MICROSLEEP_DURATION = 1.5     # detik mata tertutup -> alert (kamera sendirian)
# EMA_ALPHA           = 0.45    # smoothing skor
# PROFILE_PATH        = "calib_profile_lda.json"

# ═════════════════════════════════════════════
#  KONFIGURASI — GYRO + FUSI
# ═════════════════════════════════════════════
# GYRO_IP   = DEVICE_IP         # gyro: perangkat yang SAMA dengan kamera
# GYRO_PORT = 1235

# ═════════════════════════════════════════════
#  KETAHANAN DETEKSI WAJAH  (lensa wide / mata sipit)
# ═════════════════════════════════════════════
# Lensa wide menimbulkan distorsi barrel + vignetting, dan saat mata
# menutup penanda visual berkurang -> MediaPipe kadang kehilangan wajah.
#
# GRACE PERIOD: kalau wajah hilang SEBENTAR sementara mata sedang
# tertutup, JANGAN reset timer. Kehilangan wajah tepat saat mata
# menutup justru gejala khas microsleep — kalau di-reset, alarm tidak
# akan pernah bunyi. Di luar jendela ini, state tetap di-reset supaya
# menoleh/kamera terhalang tidak memicu alarm palsu.
# FACE_LOSS_GRACE = 0.7        # detik

# Ambang kepercayaan MediaPipe. Diturunkan karena wajah terdistorsi
# lensa wide lebih sulit dikenali daripada wajah dari lensa normal.
# DETECT_CONFIDENCE  = 0.15
# PRESENCE_CONFIDENCE = 0.15
# TRACKING_CONFIDENCE = 0.15

# CLAHE: perataan kontras lokal untuk membantu MediaPipe mengunci wajah
# saat pencahayaan tidak merata (dahi terang, mata gelap).
# PENTING: hanya dipakai untuk INPUT DETEKSI. Ekstraksi fitur mata tetap
# memakai frame ASLI, supaya nilai fitur (dark_ratio, contrast, lap_var)
# tidak bergeser dan profil kalibrasi tetap konsisten.
# USE_CLAHE = True

# Setelah sekian frame gagal berturut-turut, coba deteksi ulang memakai
# detektor mode IMAGE (deteksi penuh, bukan pelacakan). Berguna saat
# pelacakan mode VIDEO tersangkut dan tidak bisa mengunci ulang.
# FALLBACK_AFTER_MISSES = 3

# HEAD_PROFILE_PATH = "calib_profile_head.json"

# Data gyro dianggap basi kalau lebih lama dari ini.
# Dilonggarkan ke 2.5s karena link WiFi hotspot (terutama hotspot HP)
# sering mengirim data bergerombol dengan jeda 1-2 detik. Nilai ketat
# seperti 1.0s membuat gyro terus-menerus dianggap "terputus" padahal
# koneksinya sebenarnya hidup.
# GYRO_MAX_AGE = 2.5

# Durasi untuk JALUR B (kepala menunduk bertahan + mata tertutup)
# GYRO_FUSED_EYE_DURATION  = 0.8
# GYRO_FUSED_HEAD_DURATION = 0.8

# JALUR C (hentakan kepala): mata tertutup sesingkat ini sudah cukup
# asalkan terjadi hentakan dalam jendela waktu di bawah.
# JERK_EYE_DURATION   = 0.5
# JERK_WINDOW         = 1.5    # detik; hentakan dianggap relevan selama ini

# Nilai default kalau kalibrasi kepala dilewati (dipakai apa adanya)
# DEFAULT_PITCH_THRESHOLD = 25.0
# (DEFAULT_RATE_THRESHOLD dihapus - deteksi angguk kini pakai LDA)

# MODEL_PATH = "face_landmarker.task"
# MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
#               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

# Landmark EAR
# LEFT_EYE  = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Sudut mata (untuk menentukan ROI yang STABIL, tidak ikut mengecil
# saat mata tertutup — ini penting supaya ROI terbuka vs tertutup
# menutupi area yang sama dan bisa dibandingkan)
# LEFT_CORNERS  = (33, 133)     # (luar, dalam)
# RIGHT_CORNERS = (362, 263)    # (dalam, luar)

# FEATURE_NAMES = [
#     "ear",           # geometri: eye aspect ratio
#     "open_norm",     # geometri: bukaan vertikal / jarak antar mata
#     "dark_ratio",    # tampilan: proporsi piksel gelap (pupil/iris)
#     "contrast",      # tampilan: kontras gelap-terang dalam ROI
#     "lap_var",       # tampilan: ketajaman tekstur (iris punya detail)
#     "dark_y_spread", # tampilan: sebaran vertikal piksel gelap
#                      #   pupil = blob bulat  -> sebaran besar
#                      #   bulu mata = garis   -> sebaran kecil
# ]
# NF = len(FEATURE_NAMES)


# ═════════════════════════════════════════════
#  FASE KALIBRASI
# ═════════════════════════════════════════════
# PHASES = [
#     {"key": "neutral", "kind": "open", "durasi": 5.0,
#      "title": "1/6  PANDANGAN NETRAL",
#      "instruksi": ["Duduk seperti posisi mengemudi normal.",
#                    "Lihat LURUS ke depan, rileks.",
#                    "Boleh berkedip biasa."]},

#     {"key": "wide", "kind": "open", "durasi": 3.5,
#      "title": "2/6  MATA TERBUKA LEBAR",
#      "instruksi": ["Buka mata SELEBAR mungkin.",
#                    "Tahan, usahakan tidak berkedip."]},

#     {"key": "down", "kind": "open", "durasi": 5.0,
#      "title": "3/6  PANDANGAN KE BAWAH",
#      "instruksi": ["Kepala TETAP TEGAK menghadap depan.",
#                    "Turunkan hanya BOLA MATA ke bawah",
#                    "(seperti melihat speedometer).",
#                    "PENTING: mata harus tetap TERBUKA."]},

#     {"key": "squint", "kind": "open", "durasi": 4.0,
#      "title": "4/6  MATA DISIPITKAN",
#      "instruksi": ["Sipitkan mata seperti kena silau.",
#                    "Mata TETAP terbuka, jangan tertutup penuh."]},

#     {"key": "blink", "kind": "blink", "durasi": 5.0,
#      "title": "5/6  KEDIPAN NORMAL",
#      "instruksi": ["Lihat lurus ke depan.",
#                    "Berkedip normal beberapa kali.",
#                    "(fase validasi, tidak dipakai hitung ambang)"]},

#     {"key": "closed", "kind": "closed", "durasi": 5.0,
#      "title": "6/6  MATA TERTUTUP",
#      "instruksi": ["TUTUP mata dengan RILEKS,",
#                    "seperti saat mengantuk / tertidur.",
#                    "Jangan dipejamkan kuat-kuat."]},
# ]

# SETTLE_TIME = 1.2


# ═════════════════════════════════════════════
#  KALIBRASI DIPANDU BUZZER (tanpa tombol spasi)
#
#  Alur satu fase:
#    [instruksi ditampilkan]  -> PREP_TIME detik
#    beep .. beep .. BEEP     -> COUNTDOWN (mulai rekam saat beep panjang)
#    [merekam]                -> SETTLE_TIME + durasi fase
#    beeeep panjang           -> fase selesai
#
#  PENTING: nilai di bawah HARUS sama dengan pola di firmware .ino,
#  supaya perekaman dimulai tepat saat beep panjang berbunyi.
# ═════════════════════════════════════════════
# BUZZ_BOOT     = "B"    # melodi mulai kalibrasi (~1.0 s)
# BUZZ_COUNT    = "C"    # hitung mundur 3..2..1  (2.6 s)
# BUZZ_END      = "E"    # fase selesai           (0.8 s)
# BUZZ_SUCCESS  = "S"    # kalibrasi berhasil     (~1.4 s)
# BUZZ_FAIL     = "F"    # kalibrasi gagal        (1.0 s)

# BOOT_MELODY_TIME = 1.0     # durasi melodi 'B'
# HAT_ADJUST_TIME  = 10.0    # jeda membenahi posisi topi (sebelum fase 1)
# PREP_TIME        = 3.0     # jeda membaca instruksi antar fase
# COUNTDOWN_LEAD   = 2.0     # dari 'C' dikirim s/d beep panjang berbunyi
# COUNTDOWN_TOTAL  = 2.6     # total durasi pola hitung mundur
# END_BEEP_TIME    = 0.8     # durasi beep penanda fase selesai


# ═════════════════════════════════════════════
#  FASE KALIBRASI KEPALA (gyro + accelerometer)
#
#  kind "normal"     -> gerakan wajar orang sadar, TIDAK boleh memicu alarm
#  kind "microsleep" -> gerakan khas terkantuk, HARUS memicu alarm
#
#  Kunci pemisahnya:
#    - pitch  memisahkan kepala TEGAK dari kepala MENUNDUK
#    - rate   memisahkan gerakan SADAR yang pelan dari kepala
#             TERKANTUK yang jatuh cepat tak terkendali
# ═════════════════════════════════════════════
# HEAD_PHASES = [
#     {"key": "h_neutral", "kind": "normal", "durasi": 5.0,
#      "title": "1/6  KEPALA TEGAK NORMAL",
#      "instruksi": ["Duduk seperti sedang menyetir.",
#                    "Kepala tegak, pandangan ke depan.",
#                    "Gerakan kecil wajar tidak apa-apa."]},

#     {"key": "h_yaw", "kind": "normal", "durasi": 6.0,
#      "title": "2/6  MENOLEH KIRI-KANAN",
#      "instruksi": ["Tengok kiri, lalu kanan, berulang agak cepat.",
#                    "Seperti mengecek spion / blind spot.",
#                    "Kepala JANGAN menunduk, hanya menoleh."]},

#     {"key": "h_lookdown", "kind": "normal", "durasi": 6.0,
#      "title": "3/6  LIHAT SPEEDOMETER (SADAR)",
#      "instruksi": ["Tundukkan kepala PELAN ke dashboard,",
#                    "tahan sebentar, lalu angkat lagi. Ulangi.",
#                    "Gerakan SADAR dan TERKENDALI.",
#                    "Mata tetap TERBUKA."]},

#     {"key": "h_vibration", "kind": "normal", "durasi": 5.0,
#      "title": "4/6  GUNCANGAN JALAN",
#      "instruksi": ["Goyangkan topi/kepala naik-turun kecil,",
#                    "seperti melewati jalan bergelombang.",
#                    "Kepala tetap TEGAK, hanya bergetar."]},

#     {"key": "h_nod", "kind": "microsleep", "durasi": 9.0,
#      "title": "5/6  SIMULASI TERKANTUK (HENTAKAN)",
#      "instruksi": ["Jatuhkan kepala ke depan CEPAT,",
#                    "lalu sentak balik ke atas (kaget).",
#                    "Hentak oleng kanan dan kiri juga",
#                    "Ulangi 5-6 kali selama perekaman."]},

#     {"key": "h_slump", "kind": "microsleep", "durasi": 5.0,
#      "title": "6/6  KEPALA TERKULAI DIAM",
#      "instruksi": ["Biarkan kepala jatuh ke depan",
#                    "lalu DIAM di bawah, rileks total.",
#                    "Seperti orang benar-benar tertidur."]},
# ]


# ═════════════════════════════════════════════
#  UTILITAS DASAR
# ═════════════════════════════════════════════
# def ensure_model():
#     if not os.path.exists(MODEL_PATH):
#         print("Mengunduh model face_landmarker.task (sekali saja) ...")
#         urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
#         print("Selesai.\n")


# def recv_all(sock, size):
#     data = b''
#     while len(data) < size:
#         packet = sock.recv(size - len(data))
#         if not packet:
#             return None
#         data += packet
#     return data


# ═════════════════════════════════════════════
#  BANTUAN DETEKSI WAJAH UNTUK LENSA WIDE
# ═════════════════════════════════════════════
# _clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


# def enhance_for_detection(gray):
#     """
#     Perataan kontras lokal (CLAHE) untuk membantu MediaPipe mengunci wajah.
#     HANYA dipakai sebagai input deteksi — ekstraksi fitur mata tetap
#     memakai frame ASLI supaya nilai fitur tidak bergeser.
#     """
#     if not USE_CLAHE:
#         return gray
#     return _clahe.apply(gray)


# def detect_face(detector, detector_img, gray, ts, miss_count):
#     """
#     Deteksi wajah dengan dua lapis ketahanan:
#       1. Mode VIDEO (cepat, pakai pelacakan antar frame)
#       2. Kalau gagal berturut-turut -> mode IMAGE (deteksi penuh dari nol)

#     Return: (landmarks | None, miss_count_baru)
#     """
#     enhanced = enhance_for_detection(gray)
#     rgb = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
#                        cv2.COLOR_BGR2RGB)
#     mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     ts[0] += 33
#     res = detector.detect_for_video(mp_img, ts[0])
#     if res.face_landmarks:
#         return res.face_landmarks[0], 0

#     miss_count += 1

#     # Pelacakan VIDEO gagal -> coba deteksi penuh mode IMAGE
#     if detector_img is not None and miss_count >= FALLBACK_AFTER_MISSES:
#         res2 = detector_img.detect(mp_img)
#         if res2.face_landmarks:
#             return res2.face_landmarks[0], 0

#     return None, miss_count


# # ═════════════════════════════════════════════
# #  LAYAR OTOMATIS UNTUK KALIBRASI DIPANDU BUZZER
# #  (menggantikan penungguan tombol SPASI)
# # ═════════════════════════════════════════════
# def screen_prep(sock, phase, seconds, gyro_state_fn=None, sub=""):
#     """
#     Tampilkan instruksi fase selama `seconds` detik, lalu lanjut OTOMATIS.
#     Kamera tetap ditampilkan supaya user bisa membenahi posisi.
#     Return False kalau user menekan ESC (batal).
#     """
#     t0 = time.time()
#     while True:
#         sisa = seconds - (time.time() - t0)
#         if sisa <= 0:
#             return True

#         frame = read_frame(sock)
#         if frame is None:
#             return False
#         disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         h, w = frame.shape

#         draw_panel(disp, 0, 152)
#         cv2.putText(disp, phase["title"], (12, 26), cv2.FONT_HERSHEY_DUPLEX,
#                     0.60, (0, 220, 255), 1, cv2.LINE_AA)
#         draw_text_block(disp, phase["instruksi"], 12, 52)

#         draw_panel(disp, h - 56, h)
#         cv2.putText(disp, f"Bersiap... mulai dalam {sisa:0.1f}s",
#                     (12, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
#                     (0, 255, 120), 1, cv2.LINE_AA)
#         if sub:
#             cv2.putText(disp, sub, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.42, (170, 170, 170), 1, cv2.LINE_AA)
#         else:
#             cv2.putText(disp, "Dengarkan buzzer  |  ESC = batal",
#                         (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
#                         (170, 170, 170), 1, cv2.LINE_AA)

#         # Bar mundur
#         p = 1.0 - max(sisa / max(seconds, 1e-6), 0.0)
#         cv2.rectangle(disp, (0, h - 6), (int(w * p), h), (0, 200, 120), -1)

#         cv2.imshow("Microsleep Detector", disp)
#         if (cv2.waitKey(1) & 0xFF) == 27:
#             return False


# def screen_countdown(sock, phase, seconds=COUNTDOWN_LEAD):
#     """
#     Layar hitung mundur 3..2..1 yang SELARAS dengan beep di ESP32.
#     Buzzer sudah dipicu sebelum fungsi ini dipanggil.
#     Return False kalau ESC.
#     """
#     t0 = time.time()
#     while True:
#         elapsed = time.time() - t0
#         if elapsed >= seconds:
#             return True

#         frame = read_frame(sock)
#         if frame is None:
#             return False
#         disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         h, w = frame.shape

#         # 0-1s -> "3", 1-2s -> "2", lalu beep panjang = MULAI
#         angka = 3 - int(elapsed)
#         angka = max(angka, 1)

#         draw_panel(disp, 0, 60)
#         cv2.putText(disp, phase["title"], (12, 24), cv2.FONT_HERSHEY_DUPLEX,
#                     0.56, (0, 220, 255), 1, cv2.LINE_AA)
#         cv2.putText(disp, "Bersiap...", (12, 48), cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5, (200, 200, 200), 1, cv2.LINE_AA)
#         cv2.putText(disp, str(angka), (w // 2 - 26, h // 2 + 26),
#                     cv2.FONT_HERSHEY_DUPLEX, 2.4, (0, 220, 255), 4, cv2.LINE_AA)

#         cv2.imshow("Microsleep Detector", disp)
#         if (cv2.waitKey(1) & 0xFF) == 27:
#             return False


# def screen_wait(sock, seconds, judul, baris, warna=(0, 220, 255)):
#     """Layar tunggu umum (mis. jeda membenahi topi di awal)."""
#     t0 = time.time()
#     while True:
#         sisa = seconds - (time.time() - t0)
#         if sisa <= 0:
#             return True
#         frame = read_frame(sock)
#         if frame is None:
#             return False
#         disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         h, w = frame.shape

#         draw_panel(disp, 0, 140)
#         cv2.putText(disp, judul, (12, 26), cv2.FONT_HERSHEY_DUPLEX,
#                     0.60, warna, 1, cv2.LINE_AA)
#         draw_text_block(disp, baris, 12, 52)

#         draw_panel(disp, h - 42, h)
#         cv2.putText(disp, f"{sisa:0.1f} detik", (12, h - 14),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 2, cv2.LINE_AA)
#         p = 1.0 - max(sisa / max(seconds, 1e-6), 0.0)
#         cv2.rectangle(disp, (0, h - 6), (int(w * p), h), warna, -1)

#         cv2.imshow("Microsleep Detector", disp)
#         if (cv2.waitKey(1) & 0xFF) == 27:
#             return False


# def read_frame(sock):
#     size_data = recv_all(sock, 4)
#     if size_data is None:
#         return None
#     frame_size = struct.unpack("<I", size_data)[0]
#     jpg = recv_all(sock, frame_size)
#     if jpg is None:
#         return None
#     f = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#     if f is None:
#         return None
#     return cv2.flip(f, 0)     # kamera topi terbalik


# def euclidean(p1, p2):
#     return float(np.linalg.norm(np.array(p1) - np.array(p2)))


# def lm_xy(landmarks, idx, w, h):
#     lm = landmarks[idx]
#     return (lm.x * w, lm.y * h)


# def draw_panel(img, y0, y1, alpha=0.6):
#     ov = img.copy()
#     cv2.rectangle(ov, (0, y0), (img.shape[1], y1), (0, 0, 0), -1)
#     cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


# def draw_text_block(img, lines, x, y, scale=0.5, color=(230, 230, 230),
#                     gap=21):
#     for i, ln in enumerate(lines):
#         cv2.putText(img, ln, (x, y + i * gap), cv2.FONT_HERSHEY_SIMPLEX,
#                     scale, color, 1, cv2.LINE_AA)


# ═════════════════════════════════════════════
#  EKSTRAKSI FITUR
# ═════════════════════════════════════════════
# def eye_roi_box(landmarks, corners, w, h):
#     """
#     Kotak ROI mata yang STABIL — ukurannya ditentukan oleh JARAK SUDUT MATA
#     (yang tidak berubah saat mata menutup), bukan oleh tinggi bukaan mata.
#     Dengan begitu ROI mata terbuka & tertutup mencakup area yang sama.
#     """
#     p1 = lm_xy(landmarks, corners[0], w, h)
#     p2 = lm_xy(landmarks, corners[1], w, h)
#     cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
#     eye_w = euclidean(p1, p2)
#     if eye_w < 6:
#         return None

#     bw = eye_w * 1.15
#     bh = eye_w * 0.72

#     x0 = int(round(cx - bw / 2)); x1 = int(round(cx + bw / 2))
#     y0 = int(round(cy - bh / 2)); y1 = int(round(cy + bh / 2))

#     x0 = max(0, x0); y0 = max(0, y0)
#     x1 = min(w - 1, x1); y1 = min(h - 1, y1)
#     if x1 - x0 < 8 or y1 - y0 < 6:
#         return None
#     return (x0, y0, x1, y1)


# def appearance_features(gray, box):
#     """
#     Fitur berbasis tampilan piksel di dalam ROI mata.
#     Semua dinormalisasi terhadap kecerahan kulit (persentil 90)
#     sehingga tahan terhadap perubahan pencahayaan.
#     """
#     x0, y0, x1, y1 = box
#     roi = gray[y0:y1, x0:x1]
#     if roi.size < 48:
#         return None

#     roi = cv2.GaussianBlur(roi, (3, 3), 0).astype(np.float32)

#     p90 = float(np.percentile(roi, 90))
#     p05 = float(np.percentile(roi, 5))
#     denom = p90 + 1.0

#     norm = roi / denom                    # 1.0 = kulit terang
#     contrast = 1.0 - (p05 / denom)        # besar = ada area sangat gelap

#     dark_mask = norm < 0.58
#     dark_ratio = float(dark_mask.mean())

#     # Sebaran vertikal piksel gelap:
#     #   pupil (blob bulat)      -> sebaran besar
#     #   garis bulu mata (tipis) -> sebaran kecil
#     ys, xs = np.nonzero(dark_mask)
#     rh = roi.shape[0]
#     if len(ys) >= 6 and rh > 1:
#         dark_y_spread = float(np.std(ys) / rh)
#     else:
#         dark_y_spread = 0.0

#     # Tekstur: iris & pantulan cahaya menghasilkan tepi tajam
#     eq = cv2.equalizeHist(gray[y0:y1, x0:x1])
#     lap_var = float(cv2.Laplacian(eq, cv2.CV_64F).var() / 1000.0)

#     return dark_ratio, contrast, lap_var, dark_y_spread


# def eye_aspect_ratio(landmarks, eye_idx, w, h):
#     pts = [lm_xy(landmarks, i, w, h) for i in eye_idx]
#     v1 = euclidean(pts[1], pts[5])
#     v2 = euclidean(pts[2], pts[4])
#     hz = euclidean(pts[0], pts[3])
#     if hz == 0:
#         return None
#     return (v1 + v2) / (2.0 * hz)


# def extract_features(landmarks, gray, w, h):
#     """Kembalikan vektor fitur (NF,) atau None, plus kotak ROI untuk display."""
#     ear_l = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
#     ear_r = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
#     if ear_l is None or ear_r is None:
#         return None, None
#     ear = (ear_l + ear_r) / 2.0

#     # Bukaan vertikal dinormalisasi jarak antar-mata (skala wajah)
#     iod = euclidean(lm_xy(landmarks, 33, w, h), lm_xy(landmarks, 263, w, h))
#     if iod < 10:
#         return None, None
#     open_l = euclidean(lm_xy(landmarks, 159, w, h), lm_xy(landmarks, 145, w, h))
#     open_r = euclidean(lm_xy(landmarks, 386, w, h), lm_xy(landmarks, 374, w, h))
#     open_norm = ((open_l + open_r) / 2.0) / iod

#     box_l = eye_roi_box(landmarks, LEFT_CORNERS,  w, h)
#     box_r = eye_roi_box(landmarks, RIGHT_CORNERS, w, h)
#     if box_l is None or box_r is None:
#         return None, None

#     ap_l = appearance_features(gray, box_l)
#     ap_r = appearance_features(gray, box_r)
#     if ap_l is None or ap_r is None:
#         return None, None

#     dark_ratio    = (ap_l[0] + ap_r[0]) / 2.0
#     contrast      = (ap_l[1] + ap_r[1]) / 2.0
#     lap_var       = (ap_l[2] + ap_r[2]) / 2.0
#     dark_y_spread = (ap_l[3] + ap_r[3]) / 2.0

#     vec = np.array([ear, open_norm, dark_ratio, contrast, lap_var,
#                     dark_y_spread], dtype=np.float64)
#     return vec, (box_l, box_r)


# ═════════════════════════════════════════════
#  STATISTIK ROBUST
# ═════════════════════════════════════════════
# def robust_low(vals, drop=0.20):
#     """Nilai terendah setelah membuang `drop` bagian terbawah."""
#     s = np.sort(np.asarray(vals, dtype=float))
#     if len(s) < 5:
#         return float(s.min())
#     return float(s[int(len(s) * drop)])


# def robust_high(vals, drop=0.20):
#     """Nilai tertinggi setelah membuang `drop` bagian teratas."""
#     s = np.sort(np.asarray(vals, dtype=float))
#     if len(s) < 5:
#         return float(s.max())
#     return float(s[len(s) - 1 - int(len(s) * drop)])


# ═════════════════════════════════════════════
#  PEREKAMAN FASE KALIBRASI
# ═════════════════════════════════════════════
# def record_phase(sock, detector, detector_img, ts, phase, gyro=None,
#                  prep_time=PREP_TIME):
#     key    = phase["key"]
#     durasi = phase["durasi"]
#     feats  = []
#     cal_miss = 0

#     # ── 1. Instruksi (lanjut OTOMATIS, tanpa tombol) ──
#     if not screen_prep(sock, phase, prep_time):
#         return None

#     # ── 2. Hitung mundur, diselaraskan dengan beep di ESP32 ──
#     if gyro is not None:
#         gyro.buzz(BUZZ_COUNT)          # beep .. beep .. BEEP-panjang
#     if not screen_countdown(sock, phase, COUNTDOWN_LEAD):
#         return None
#     # Beep panjang berbunyi TEPAT sekarang = aba-aba "MULAI"

#     # ── Perekaman ──
#     total = SETTLE_TIME + durasi
#     t0 = time.time()
#     while True:
#         elapsed = time.time() - t0
#         if elapsed >= total:
#             break

#         frame = read_frame(sock)
#         if frame is None:
#             return None
#         h, w = frame.shape

#         bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#         # Deteksi dengan CLAHE + fallback IMAGE, sama seperti mode deteksi,
#         # supaya fase "mata tertutup" tidak kehilangan banyak sampel.
#         lms, cal_miss = detect_face(detector, detector_img, frame, ts, cal_miss)

#         disp = bgr.copy()
#         vec = None

#         if lms is not None:
#             vec, boxes = extract_features(lms, frame, w, h)
#             if boxes is not None:
#                 for (x0, y0, x1, y1) in boxes:
#                     cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 200, 255), 1)
#             for idx in LEFT_EYE + RIGHT_EYE:
#                 lm = lms[idx]
#                 cv2.circle(disp, (int(lm.x * w), int(lm.y * h)), 2,
#                            (0, 255, 0), -1)

#         recording = elapsed >= SETTLE_TIME
#         if recording and vec is not None:
#             feats.append(vec)

#         draw_panel(disp, 0, 80)
#         cv2.putText(disp, phase["title"], (12, 24), cv2.FONT_HERSHEY_DUPLEX,
#                     0.58, (0, 220, 255), 1, cv2.LINE_AA)
#         if recording:
#             cv2.putText(disp, f"MEREKAM  sisa {total - elapsed:0.1f}s  n={len(feats)}",
#                         (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
#                         (0, 80, 255), 2, cv2.LINE_AA)
#             cv2.circle(disp, (w - 22, 22), 8, (0, 80, 255), -1)
#         else:
#             cv2.putText(disp, "bersiap (belum merekam)", (12, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1,
#                         cv2.LINE_AA)
#         if vec is not None:
#             cv2.putText(disp, f"EAR {vec[0]:.3f}  dark {vec[2]:.2f}  spr {vec[5]:.2f}",
#                         (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
#                         (0, 255, 120), 1, cv2.LINE_AA)
#         else:
#             cv2.putText(disp, "wajah tidak terdeteksi", (12, 72),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1,
#                         cv2.LINE_AA)

#         cv2.rectangle(disp, (0, h - 6), (int(w * min(elapsed / total, 1.0)), h),
#                       (0, 220, 255), -1)
#         cv2.imshow("Microsleep Detector", disp)
#         if (cv2.waitKey(1) & 0xFF) == 27:
#             return None

#     # ── 4. Beep panjang: fase selesai ──
#     if gyro is not None:
#         gyro.buzz(BUZZ_END)

#     if len(feats) < 8:
#         print(f"  [!] Fase '{key}': hanya {len(feats)} sampel valid. Terlalu sedikit.")
#         return None

#     return np.vstack(feats)


# def run_calibration(sock, detector, detector_img, ts, gyro=None,
#                     first_prep=None):
#     """
#     6 fase kalibrasi MATA, dipandu buzzer, tanpa tombol.
#     first_prep: lama jeda sebelum fase PERTAMA (dipakai untuk jeda
#                 membenahi topi di awal rangkaian 12 fase).
#     """
#     print("\n" + "=" * 60)
#     print("  FASE KALIBRASI MATA  (6 tahap)")
#     print("=" * 60)

#     data = {}
#     for i, phase in enumerate(PHASES):
#         prep = first_prep if (i == 0 and first_prep is not None) else PREP_TIME
#         arr = record_phase(sock, detector, detector_img, ts, phase,
#                            gyro=gyro, prep_time=prep)
#         if arr is None:
#             return None
#         data[phase["key"]] = {"kind": phase["kind"], "X": arr}
#         print(f"  {phase['title']:<26} n={len(arr):3d}   "
#               f"EAR mean={arr[:, 0].mean():.3f}   "
#               f"dark={arr[:, 2].mean():.3f}   spread={arr[:, 5].mean():.3f}")
#     return data


# ═════════════════════════════════════════════
#  ANALISIS + LDA
# ═════════════════════════════════════════════
# def separability(a, b):
#     """d-prime: seberapa terpisah dua distribusi 1-D."""
#     va, vb = np.var(a), np.var(b)
#     denom = np.sqrt(0.5 * (va + vb)) + 1e-9
#     return abs(np.mean(a) - np.mean(b)) / denom


# def fit_lda(X_awake, X_closed):
#     """
#     Fisher LDA: cari arah w yang paling memisahkan kelas terjaga vs tertutup.
#     Fitur distandardisasi dulu agar skala tidak mendominasi.
#     """
#     X_all = np.vstack([X_awake, X_closed])
#     mu = X_all.mean(axis=0)
#     sd = X_all.std(axis=0) + 1e-9

#     A = (X_awake  - mu) / sd
#     C = (X_closed - mu) / sd

#     mA, mC = A.mean(axis=0), C.mean(axis=0)
#     Sw = np.cov(A, rowvar=False) * len(A) + np.cov(C, rowvar=False) * len(C)
#     Sw /= (len(A) + len(C))
#     Sw += np.eye(NF) * 1e-3          # regularisasi

#     w = np.linalg.pinv(Sw) @ (mA - mC)
#     n = np.linalg.norm(w)
#     if n < 1e-9:
#         return None
#     w = w / n

#     # Pastikan arah positif = mata terbuka
#     if (A @ w).mean() < (C @ w).mean():
#         w = -w

#     return {"w": w, "mu": mu, "sd": sd}


# def project(model, X):
#     return ((np.atleast_2d(X) - model["mu"]) / model["sd"]) @ model["w"]


# def analyze(data, exclude_open_phases=()):
#     open_keys = [k for k, v in data.items()
#                  if v["kind"] == "open" and k not in exclude_open_phases]
#     if not open_keys:
#         return None

#     X_awake  = np.vstack([data[k]["X"] for k in open_keys])
#     X_closed = data["closed"]["X"]

#     print("\n" + "-" * 60)
#     print("  DAYA PISAH TIAP FITUR  (d-prime, makin besar makin bagus)")
#     print("-" * 60)
#     for i, name in enumerate(FEATURE_NAMES):
#         d = separability(X_awake[:, i], X_closed[:, i])
#         bar = "#" * min(int(d * 4), 34)
#         print(f"  {name:<14} d'={d:5.2f}  {bar}")

#     # Fitur mana yang berhasil memisahkan MENUNDUK dari TERTUTUP?
#     if "down" in data and "down" not in exclude_open_phases:
#         print("\n  Khusus MENUNDUK vs TERTUTUP (kasus tersulit):")
#         for i, name in enumerate(FEATURE_NAMES):
#             d = separability(data["down"]["X"][:, i], X_closed[:, i])
#             flag = "  <== penyelamat" if d > 1.2 else ""
#             print(f"    {name:<14} d'={d:5.2f}{flag}")

#     model = fit_lda(X_awake, X_closed)
#     if model is None:
#         return None

#     print("\n  Bobot LDA hasil kalibrasi:")
#     for name, wi in zip(FEATURE_NAMES, model["w"]):
#         print(f"    {name:<14} {wi:+.3f}")

#     # ── Ambang dari skor terproyeksi ──
#     floors = {k: robust_low(project(model, data[k]["X"])) for k in open_keys}
#     awake_floor = min(floors.values())
#     critical    = min(floors, key=floors.get)
#     closed_ceil = robust_high(project(model, X_closed))
#     gap = awake_floor - closed_ceil

#     print("\n" + "-" * 60)
#     print("  HASIL PEMISAHAN (skor LDA)")
#     print("-" * 60)
#     for k, v in floors.items():
#         mark = "  <-- paling kritis" if k == critical else ""
#         print(f"  Lantai skor terbuka [{k:<8}] : {v:+.3f}{mark}")
#     print(f"  Plafon skor tertutup          : {closed_ceil:+.3f}")
#     print(f"  Gap pemisah                   : {gap:+.3f}")

#     if gap <= 0.05:
#         print(f"\n  [X] GAGAL: kelas masih tumpang tindih. Biang keroknya: '{critical}'.")
#         return {"failed": True, "critical": critical, "gap": gap}

#     t_close = closed_ceil + 0.40 * gap
#     t_open  = closed_ceil + 0.62 * gap

#     blink = project(model, data["blink"]["X"])
#     dips  = int(np.sum(blink < t_close))
#     ratio = dips / len(blink)

#     print(f"\n  Ambang TUTUP (t_close)        : {t_close:+.3f}")
#     print(f"  Ambang BUKA  (t_open)         : {t_open:+.3f}")
#     print(f"  Kedipan menembus ambang       : {dips}/{len(blink)} ({ratio*100:.0f}%)")

#     if ratio > 0.55:
#         print("  [!] Kedipan terlalu sering di bawah ambang — mungkin ambang kelewat tinggi.")
#     elif dips == 0:
#         print("  [OK] Kedipan cepat tidak menembus ambang. Normal, tidak masalah.")
#     else:
#         print("  [OK] Kedipan terdeteksi wajar. Ambang sehat.")

#     if gap < 0.35:
#         print("  [!] Gap agak sempit. Deteksi bisa sensitif terhadap noise.")

#     print("-" * 60 + "\n")

#     return {
#         "failed":   False,
#         "w":        model["w"].tolist(),
#         "mu":       model["mu"].tolist(),
#         "sd":       model["sd"].tolist(),
#         "t_close":  float(t_close),
#         "t_open":   float(t_open),
#         "gap":      float(gap),
#         "awake_floor":    float(awake_floor),
#         "closed_ceiling": float(closed_ceil),
#         "critical_phase": critical,
#         "excluded":       list(exclude_open_phases),
#         "features": FEATURE_NAMES,
#         "waktu":    time.strftime("%Y-%m-%d %H:%M:%S"),
#     }


# def build_profile(data):
#     """Coba pakai semua fase. Kalau gagal, tawarkan mengeluarkan fase biang kerok."""
#     prof = analyze(data)
#     if prof is None:
#         return None
#     if not prof.get("failed"):
#         return prof

#     culprit = prof["critical"]
#     print("\n" + "!" * 60)
#     print(f"  Fase '{culprit}' tidak bisa dipisahkan dari kondisi mata tertutup")
#     print("  pada posisi kamera ini — secara fisik kedua kondisi memang")
#     print("  terlihat nyaris identik dari sudut pandang kamera.")
#     print("\n  Dua pilihan:")
#     print(f"    [1] Keluarkan fase '{culprit}' dari kalibrasi.")
#     print(f"        Konsekuensi: bertahan di posisi '{culprit}' lebih dari")
#     print(f"        {MICROSLEEP_DURATION:.1f} detik akan ikut memicu alarm.")
#     print("    [2] Batalkan, perbaiki posisi kamera, lalu kalibrasi ulang.")
#     print("        (miringkan kamera agar mata terlihat lebih dari depan)")
#     print("!" * 60)

#     ans = input(f"\nKeluarkan fase '{culprit}'? [y/N] ").strip().lower()
#     if ans != "y":
#         return None

#     prof2 = analyze(data, exclude_open_phases=(culprit,))
#     if prof2 is None or prof2.get("failed"):
#         print("\n  Masih gagal juga. Posisi kamera perlu diperbaiki.")
#         return None

#     print(f"\n  [OK] Kalibrasi berhasil tanpa fase '{culprit}'.")
#     print(f"  INGAT: kondisi '{culprit}' sekarang dianggap mata tertutup.\n")
#     return prof2


# ═════════════════════════════════════════════
#  KALIBRASI KEPALA (gyro)
# ═════════════════════════════════════════════
# def record_head_phase(sock, gyro, phase, prep_time=PREP_TIME):
#     pitches, feats = [], []

#     # Buang sampel lama supaya tidak tercampur fase sebelumnya
#     gyro.drain_samples()

#     # ── 1. Instruksi (lanjut OTOMATIS, tanpa tombol) ──
#     st = gyro.get_state()
#     ok0 = st["connected"] and st["age"] is not None and st["age"] < GYRO_MAX_AGE
#     sub = ("gyro OK  pitch=%.0f  rate=%.0f" % (st["pitch"], st["rate"])
#            if ok0 else "GYRO TIDAK TERHUBUNG - data tidak akan terekam")
#     if not screen_prep(sock, phase, prep_time, sub=sub):
#         return None

#     # ── 2. Hitung mundur, diselaraskan dengan beep di ESP32 ──
#     gyro.buzz(BUZZ_COUNT)
#     if not screen_countdown(sock, phase, COUNTDOWN_LEAD):
#         return None
#     # Beep panjang berbunyi TEPAT sekarang = aba-aba "MULAI"

#     # ── Perekaman ──
#     total = SETTLE_TIME + phase["durasi"]
#     t0 = time.time()
#     while True:
#         elapsed = time.time() - t0
#         if elapsed >= total:
#             break

#         frame = read_frame(sock)
#         if frame is None:
#             return None
#         disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         h, w = frame.shape

#         g = gyro.get_state()
#         ok = g["connected"] and g["age"] is not None and g["age"] < GYRO_MAX_AGE

#         # Ambil SEMUA sampel baru (bukan hanya yang terakhir) supaya
#         # puncak angguk tidak terlewat saat data datang bergerombol
#         new_samples = gyro.drain_samples()
#         if elapsed >= SETTLE_TIME and ok:
#             for s in new_samples:
#                 pitches.append(s["pitch"])
#                 feats.append(head_feature_vector(s))

#         draw_panel(disp, 0, 80)
#         cv2.putText(disp, phase["title"], (12, 24), cv2.FONT_HERSHEY_DUPLEX,
#                     0.56, (0, 220, 255), 1, cv2.LINE_AA)
#         if elapsed >= SETTLE_TIME:
#             cv2.putText(disp, f"MEREKAM  sisa {total - elapsed:0.1f}s  n={len(pitches)}",
#                         (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
#                         (0, 80, 255), 2, cv2.LINE_AA)
#             cv2.circle(disp, (w - 22, 22), 8, (0, 80, 255), -1)
#         else:
#             cv2.putText(disp, "bersiap (belum merekam)", (12, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1,
#                         cv2.LINE_AA)
#         cv2.putText(disp, f"pitch {g['pitch']:6.1f}   prate {g['prate']:7.1f}",
#                     (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
#                     (0, 255, 120) if ok else (0, 165, 255), 1, cv2.LINE_AA)

#         cv2.rectangle(disp, (0, h - 6), (int(w * min(elapsed / total, 1.0)), h),
#                       (0, 220, 255), -1)
#         cv2.imshow("Microsleep Detector", disp)
#         if (cv2.waitKey(1) & 0xFF) == 27:
#             return None

#     # ── 4. Beep panjang: fase selesai ──
#     gyro.buzz(BUZZ_END)

#     if len(feats) < 10:
#         print(f"  [!] Fase '{phase['key']}': hanya {len(feats)} sampel. Gyro bermasalah?")
#         return None

#     return {"kind": phase["kind"],
#             "pitch": np.array(pitches),
#             "X":     np.vstack(feats)}


# def run_head_calibration(sock, gyro):
#     """6 fase kalibrasi KEPALA, dipandu buzzer, tanpa tombol."""
#     print("\n" + "=" * 60)
#     print("  KALIBRASI KEPALA (gyro + accelerometer) — 6 tahap")
#     print("=" * 60)

#     data = {}
#     for phase in HEAD_PHASES:
#         d = record_head_phase(sock, gyro, phase, prep_time=PREP_TIME)
#         if d is None:
#             return None
#         data[phase["key"]] = d
#         print(f"  {phase['title']:<32} n={len(d['pitch']):3d}  "
#               f"pitch[{d['pitch'].min():5.1f}..{d['pitch'].max():5.1f}]  "
#               f"prate_max={d['X'][:, 0].max():6.1f}")
#     return data


# def compute_head_profile(data):
#     """
#     Dua hasil:
#       pitch_threshold : memisahkan kepala TEGAK dari kepala MENUNDUK
#       model + nod_threshold : skor LDA yang memisahkan ANGGUKAN dari
#                               SEMUA gerakan sadar, termasuk MENOLEH.
#     """
#     calm_keys = ["h_neutral", "h_yaw", "h_vibration"]
#     down_keys = ["h_lookdown", "h_slump"]

#     # ── Ambang pitch (postur menunduk) ──
#     calm_pitch = np.concatenate([data[k]["pitch"] for k in calm_keys if k in data])
#     down_pitch = np.concatenate([data[k]["pitch"] for k in down_keys if k in data])

#     calm_ceiling = robust_high(calm_pitch, drop=0.05)
#     down_floor   = robust_low(down_pitch, drop=0.25)

#     print("\n" + "-" * 60)
#     print("  ANALISIS KALIBRASI KEPALA")
#     print("-" * 60)
#     print(f"  Pitch tertinggi saat tegak     : {calm_ceiling:6.1f} deg")
#     print(f"  Pitch terendah saat menunduk   : {down_floor:6.1f} deg")

#     if down_floor > calm_ceiling + 3.0:
#         pitch_threshold = calm_ceiling + 0.45 * (down_floor - calm_ceiling)
#         print(f"  -> Ambang MENUNDUK             : {pitch_threshold:6.1f} deg  [OK]")
#     else:
#         pitch_threshold = DEFAULT_PITCH_THRESHOLD
#         print(f"  -> Tumpang tindih, pakai default: {pitch_threshold:6.1f} deg  [!]")

#     # ═══════════════════════════════════════════
#     #  DETEKSI ANGGUKAN — pakai LDA, bukan ambang magnitudo
#     #
#     #  Kelas POSITIF : h_nod (angguk terkantuk)
#     #  Kelas NEGATIF : SEMUA gerakan sadar, TERMASUK MENOLEH
#     #
#     #  Dengan memasukkan h_yaw ke kelas negatif, LDA dipaksa mencari
#     #  kombinasi fitur yang membuat menoleh TIDAK terdeteksi sebagai
#     #  angguk — inilah yang tidak bisa dilakukan ambang magnitudo.
#     # ═══════════════════════════════════════════
#     aware_keys = ["h_neutral", "h_yaw", "h_lookdown", "h_vibration"]
#     X_aware = np.vstack([data[k]["X"] for k in aware_keys if k in data])
#     X_nod   = data["h_nod"]["X"] if "h_nod" in data else None

#     print("\n  DAYA PISAH FITUR (angguk vs gerakan sadar):")
#     for i, name in enumerate(HEAD_FEATURES):
#         d = separability(X_nod[:, i], X_aware[:, i]) if X_nod is not None else 0.0
#         bar = "#" * min(int(d * 4), 30)
#         print(f"    {name:<10} d'={d:5.2f}  {bar}")

#     # Cek khusus: angguk vs MENOLEH saja (kasus yang bermasalah)
#     if "h_yaw" in data and X_nod is not None:
#         print("\n  Khusus ANGGUK vs MENOLEH (kasus bermasalah):")
#         for i, name in enumerate(HEAD_FEATURES):
#             d = separability(X_nod[:, i], data["h_yaw"]["X"][:, i])
#             flag = "  <== pembeda" if d > 1.0 else ""
#             print(f"    {name:<10} d'={d:5.2f}{flag}")

#     model = fit_lda_generic(X_nod, X_aware, NHF)
#     if model is None:
#         print("\n  [X] LDA gagal. Pakai ambang default.")
#         return {"pitch_threshold": float(pitch_threshold),
#                 "nod_threshold": None, "w": None, "mu": None, "sd": None,
#                 "calm_ceiling": float(calm_ceiling),
#                 "down_floor": float(down_floor),
#                 "features": HEAD_FEATURES,
#                 "waktu": time.strftime("%Y-%m-%d %H:%M:%S")}

#     print("\n  Bobot LDA gerakan kepala:")
#     for name, wi in zip(HEAD_FEATURES, model["w"]):
#         print(f"    {name:<10} {wi:+.3f}")

#     s_nod   = head_score(model, X_nod)
#     s_aware = head_score(model, X_aware)

#     # Ambang: di atas hampir semua gerakan sadar, tapi masih di bawah
#     # puncak angguk. Condong ke sisi aman (sedikit di atas gerakan sadar).
#     aware_ceiling = float(np.percentile(s_aware, 99))
#     nod_peak      = float(np.percentile(s_nod, 85))
#     gap = nod_peak - aware_ceiling

#     print(f"\n  Skor puncak gerakan sadar      : {aware_ceiling:+.3f}")
#     print(f"  Skor puncak angguk             : {nod_peak:+.3f}")
#     print(f"  Gap pemisah                    : {gap:+.3f}")

#     if gap > 0.15:
#         nod_threshold = aware_ceiling + 0.45 * gap
#         print(f"  -> Ambang ANGGUKAN             : {nod_threshold:+.3f}  [OK]")
#     else:
#         nod_threshold = aware_ceiling + max(0.25, abs(aware_ceiling) * 0.2)
#         print(f"  -> Kurang terpisah, ambang aman: {nod_threshold:+.3f}  [!]")
#         print("     Saat fase angguk, jatuhkan kepala LEBIH CEPAT & DALAM.")

#     # ── Validasi terhadap tiap fase ──
#     print("\n  VALIDASI (berapa % frame dianggap ANGGUKAN):")
#     for key in ["h_neutral", "h_yaw", "h_lookdown", "h_vibration", "h_nod", "h_slump"]:
#         if key not in data:
#             continue
#         sc = head_score(model, data[key]["X"])
#         pct = 100.0 * np.mean(sc > nod_threshold)
#         if key == "h_nod":
#             verdict = "[OK]" if pct > 5 else "[!] angguk tidak terdeteksi"
#         elif key == "h_yaw":
#             verdict = "[OK]" if pct < 3 else "[!] MENOLEH masih salah deteksi"
#         else:
#             verdict = "[OK]" if pct < 5 else "[!] false positive"
#         print(f"    {key:<12} {pct:5.1f}%   {verdict}")

#     print("-" * 60 + "\n")

#     return {
#         "pitch_threshold": float(pitch_threshold),
#         "nod_threshold":   float(nod_threshold),
#         "w":  model["w"].tolist(),
#         "mu": model["mu"].tolist(),
#         "sd": model["sd"].tolist(),
#         "features":     HEAD_FEATURES,
#         "calm_ceiling": float(calm_ceiling),
#         "down_floor":   float(down_floor),
#         "aware_ceiling": float(aware_ceiling),
#         "nod_peak":      float(nod_peak),
#         "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
#     }


# ═════════════════════════════════════════════
#  FITUR GERAKAN KEPALA
#  Dipakai untuk membedakan ANGGUKAN (microsleep) dari
#  MENOLEH / gerakan sadar lainnya.
# ═════════════════════════════════════════════
# HEAD_FEATURES = ["abs_prate", "abs_gx", "abs_gy", "abs_gz", "accdev"]
# NHF = len(HEAD_FEATURES)


# def head_feature_vector(s):
#     """Ubah satu sampel gyro jadi vektor fitur gerakan."""
#     return np.array([
#         abs(s.get("prate", 0.0)),   # laju perubahan sudut kemiringan
#                                     #   -> BESAR saat mengangguk
#                                     #   -> ~NOL saat menoleh (kepala tetap tegak)
#         abs(s.get("gx", 0.0)),
#         abs(s.get("gy", 0.0)),
#         abs(s.get("gz", 0.0)),
#         s.get("accdev", 0.0),
#     ], dtype=np.float64)


# def fit_lda_generic(X_pos, X_neg, nf):
#     """
#     Fisher LDA umum: cari arah w yang paling memisahkan dua kelas.
#     Dipakai ulang untuk gerakan kepala (angguk vs bukan-angguk).
#     """
#     X_all = np.vstack([X_pos, X_neg])
#     mu = X_all.mean(axis=0)
#     sd = X_all.std(axis=0) + 1e-9

#     A = (X_pos - mu) / sd
#     B = (X_neg - mu) / sd

#     mA, mB = A.mean(axis=0), B.mean(axis=0)
#     Sw = np.cov(A, rowvar=False) * len(A) + np.cov(B, rowvar=False) * len(B)
#     Sw /= (len(A) + len(B))
#     Sw += np.eye(nf) * 1e-3

#     w = np.linalg.pinv(Sw) @ (mA - mB)
#     n = np.linalg.norm(w)
#     if n < 1e-9:
#         return None
#     w = w / n
#     if (A @ w).mean() < (B @ w).mean():
#         w = -w
#     return {"w": w, "mu": mu, "sd": sd}


# def head_score(model, X):
#     return ((np.atleast_2d(X) - model["mu"]) / model["sd"]) @ model["w"]


# def gyro_ready(gyro):
#     st = gyro.get_state()
#     return (st["connected"] and st["age"] is not None
#             and st["age"] < GYRO_MAX_AGE)


# def wait_for_gyro(gyro, timeout=15.0, label="Menunggu gyro"):
#     """
#     Tunggu sampai gyro benar-benar mengirim data segar.
#     GyroClient otomatis reconnect tiap 2 detik, jadi memberi waktu
#     beberapa belas detik jauh lebih baik daripada langsung menyerah.
#     """
#     print(f"{label} (maks {timeout:.0f} detik)...")
#     t0 = time.time()
#     last_dot = 0
#     while time.time() - t0 < timeout:
#         if gyro_ready(gyro):
#             st = gyro.get_state()
#             print(f"  [OK] Gyro siap. pitch={st['pitch']:.1f}  rate={st['rate']:.1f}\n")
#             return True
#         if time.time() - last_dot > 1.0:
#             print(".", end="", flush=True)
#             last_dot = time.time()
#         time.sleep(0.2)
#     print("\n  [!] Gyro belum siap.")
#     return False


# def save_head_profile(p):
#     with open(HEAD_PROFILE_PATH, "w") as f:
#         json.dump(p, f, indent=2)
#     print(f"Profil kepala disimpan: {HEAD_PROFILE_PATH}\n")


# def load_head_profile():
#     if not os.path.exists(HEAD_PROFILE_PATH):
#         return None
#     try:
#         with open(HEAD_PROFILE_PATH) as f:
#             p = json.load(f)
#         # Profil lama (format rate_threshold) tidak kompatibel -> minta ulang
#         if "pitch_threshold" in p and p.get("features") == HEAD_FEATURES:
#             return p
#         print("Profil kepala lama terdeteksi (format berbeda) -> perlu kalibrasi ulang.")
#     except Exception:
#         pass
#     return None


# def default_head_profile():
#     return {"pitch_threshold": DEFAULT_PITCH_THRESHOLD,
#             "nod_threshold": None, "w": None, "mu": None, "sd": None,
#             "features": HEAD_FEATURES,
#             "calm_ceiling": 0.0, "down_floor": 0.0,
#             "waktu": "default (belum dikalibrasi)"}


# def save_profile(p):
#     with open(PROFILE_PATH, "w") as f:
#         json.dump(p, f, indent=2)
#     print(f"Profil disimpan: {PROFILE_PATH}\n")


# def load_profile():
#     if not os.path.exists(PROFILE_PATH):
#         return None
#     try:
#         with open(PROFILE_PATH) as f:
#             p = json.load(f)
#         if all(k in p for k in ("w", "mu", "sd", "t_close", "t_open")):
#             if p.get("features") == FEATURE_NAMES:
#                 return p
#     except Exception:
#         pass
#     return None


# ═════════════════════════════════════════════
#  MODE DETEKSI
# ═════════════════════════════════════════════
# def run_detection(sock, detector, detector_img, ts, prof, gyro, head_prof):
#     w_vec = np.array(prof["w"])
#     mu    = np.array(prof["mu"])
#     sd    = np.array(prof["sd"])
#     t_close, t_open = prof["t_close"], prof["t_open"]

#     PITCH_TH = head_prof["pitch_threshold"]
#     NOD_TH   = head_prof.get("nod_threshold")
#     nod_model = None
#     if head_prof.get("w") is not None and NOD_TH is not None:
#         nod_model = {"w":  np.array(head_prof["w"]),
#                      "mu": np.array(head_prof["mu"]),
#                      "sd": np.array(head_prof["sd"])}

#     score_ema        = None
#     eyes_closed      = False
#     closed_start     = None
#     closed_duration  = 0.0
#     alert_active     = False
#     alert_reason     = ""
#     blink_count      = 0
#     show_roi         = True

#     # Pelacakan hilangnya wajah (lensa wide sering lepas kunci)
#     last_face_time   = 0.0     # kapan terakhir wajah terdeteksi
#     face_miss        = 0       # frame gagal berturut-turut
#     face_lost_for    = 999.0
#     in_grace         = False

#     # State gyro
#     head_down          = False
#     head_down_start    = None
#     head_down_duration = 0.0
#     last_jerk_time     = 0.0     # kapan terakhir hentakan terdeteksi
#     jerk_count         = 0
#     nod_score_now      = 0.0     # skor angguk terkini (untuk tampilan)

#     fps_t, fps_n, fps = time.time(), 0, 0.0

#     print("MODE DETEKSI aktif.")
#     print(f"  t_close={t_close:+.3f}   t_open={t_open:+.3f}")
#     print(f"  pitch_th={PITCH_TH:.1f}deg", end="")
#     if nod_model is not None:
#         print(f"   nod_th={NOD_TH:+.3f} (LDA aktif)")
#     else:
#         print("   deteksi angguk NONAKTIF (belum kalibrasi kepala)")
#     if prof.get("excluded"):
#         print(f"  Catatan: fase {prof['excluded']} dikeluarkan dari kalibrasi.")
#     print("  q=keluar  r=kalibrasi ulang  d=debug  e=toggle ROI\n")

#     while True:
#         frame = read_frame(sock)
#         if frame is None:
#             return "disconnect"
#         h, w = frame.shape

#         bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

#         # Deteksi wajah dengan CLAHE + fallback mode IMAGE
#         lms, face_miss = detect_face(detector, detector_img, frame, ts, face_miss)

#         disp = bgr.copy()
#         score = None
#         vec = None
#         now = time.time()

#         if lms is not None:
#             last_face_time = now
#             vec, boxes = extract_features(lms, frame, w, h)
#             if vec is not None:
#                 score = float(((vec - mu) / sd) @ w_vec)
#                 if show_roi and boxes is not None:
#                     for (x0, y0, x1, y1) in boxes:
#                         cv2.rectangle(disp, (x0, y0), (x1, y1),
#                                       (0, 200, 255), 1)
#                 for idx in LEFT_EYE + RIGHT_EYE:
#                     lm = lms[idx]
#                     cv2.circle(disp, (int(lm.x * w), int(lm.y * h)), 2,
#                                (0, 255, 0), -1)

#         if score is not None:
#             score_ema = score if score_ema is None else \
#                         EMA_ALPHA * score + (1 - EMA_ALPHA) * score_ema
#         else:
#             score_ema = None

#         # Sudah berapa lama wajah hilang
#         face_lost_for = (now - last_face_time) if last_face_time else 999.0
#         in_grace = (face_lost_for < FACE_LOSS_GRACE)

#         # ── State machine MATA (hysteresis) — TIDAK lagi memutuskan alert
#         #    sendiri, hanya melacak eyes_closed & closed_duration.
#         #    Keputusan alert dipindah ke blok FUSI di bawah. ──
#         if score_ema is not None:
#             if not eyes_closed and score_ema < t_close:
#                 eyes_closed = True
#                 closed_start = now
#             elif eyes_closed and score_ema > t_open:
#                 if 0.04 < closed_duration < MICROSLEEP_DURATION:
#                     blink_count += 1
#                 eyes_closed, closed_start = False, None
#                 closed_duration = 0.0

#             if eyes_closed and closed_start is not None:
#                 closed_duration = now - closed_start

#         elif eyes_closed and in_grace:
#             # ── GRACE PERIOD ──
#             # Wajah hilang SEBENTAR sementara mata sedang tertutup.
#             # Jangan reset timer: kehilangan wajah tepat saat mata menutup
#             # justru gejala khas microsleep (lensa wide + mata sipit membuat
#             # MediaPipe mudah lepas kunci). Timer tetap berjalan.
#             if closed_start is not None:
#                 closed_duration = now - closed_start

#         else:
#             # Wajah hilang terlalu lama, atau hilang saat mata terbuka
#             # -> reset penuh supaya menoleh / kamera terhalang tidak
#             #    memicu alarm palsu.
#             eyes_closed, closed_start = False, None
#             closed_duration = 0.0

#         # ── State GYRO: lacak durasi kepala menunduk ──
#         g = gyro.get_state()
#         gyro_ok = g["connected"] and g["age"] is not None and g["age"] < GYRO_MAX_AGE

#         if gyro_ok:
#             # Postur menunduk (bertahan)
#             if g["pitch"] > PITCH_TH:
#                 if not head_down:
#                     head_down = True
#                     head_down_start = now
#                 head_down_duration = now - head_down_start
#             else:
#                 head_down, head_down_start = False, None
#                 head_down_duration = 0.0

#             # ── Deteksi ANGGUKAN pakai skor LDA ──
#             # Diperiksa untuk SETIAP sampel yang masuk (bukan hanya yang
#             # terakhir), supaya puncak angguk tidak terlewat. LDA sudah
#             # dilatih agar MENOLEH tidak ikut terdeteksi.
#             if nod_model is not None:
#                 new_samples = gyro.drain_samples()
#                 if new_samples:
#                     X = np.vstack([head_feature_vector(s) for s in new_samples])
#                     scores = head_score(nod_model, X)
#                     nod_score_now = float(scores.max())
#                     if nod_score_now > NOD_TH:
#                         if now - last_jerk_time > 0.4:
#                             jerk_count += 1
#                         last_jerk_time = now
#         else:
#             # Gyro terputus/basi -> jangan andalkan, reset supaya tidak
#             # nyangkut di state lama begitu koneksi pulih
#             head_down, head_down_start = False, None
#             head_down_duration = 0.0

#         recent_jerk = gyro_ok and (now - last_jerk_time) < JERK_WINDOW

#         # ── FUSI KEPUTUSAN: 3 jalur, mana pun yang lebih dulu terpenuhi ──
#         # A. Kamera sendirian, threshold penuh 1.5s
#         #    -> fallback aman kalau gyro putus.
#         # B. Kamera + kepala MENUNDUK bertahan, 0.8s
#         #    -> pola tertidur perlahan, kepala terkulai.
#         # C. Kamera + HENTAKAN kepala, 0.5s
#         #    -> pola microsleep klasik: kepala jatuh lalu tersentak.
#         #       Paling cepat karena hentakan adalah sinyal paling khas.
#         path_a = eyes_closed and closed_duration >= MICROSLEEP_DURATION
#         path_b = (eyes_closed and closed_duration >= GYRO_FUSED_EYE_DURATION and
#                   gyro_ok and head_down and
#                   head_down_duration >= GYRO_FUSED_HEAD_DURATION)
#         path_c = (eyes_closed and closed_duration >= JERK_EYE_DURATION and
#                   recent_jerk)

#         should_alert = path_a or path_b or path_c

#         if should_alert and not alert_active:
#             alert_active = True
#             if path_c:
#                 alert_reason = "mata+hentakan"
#             elif path_b:
#                 alert_reason = "mata+menunduk"
#             else:
#                 alert_reason = "kamera"
#             print(f"[ALERT] MICROSLEEP! ({alert_reason})  "
#                   f"mata={closed_duration:.2f}s  menunduk={head_down_duration:.2f}s  "
#                   f"nod={nod_score_now:+.2f}")
#         elif not should_alert and alert_active:
#             alert_active = False
#             alert_reason = ""

#         gyro.set_buzzer(alert_active)

#         # ── Overlay ──
#         draw_panel(disp, 0, 78, alpha=0.55)
#         if score_ema is None:
#             if eyes_closed and in_grace:
#                 # Wajah hilang tapi timer TETAP jalan — beri tahu user
#                 cv2.putText(disp, f"TERTUTUP  {closed_duration:.1f}s",
#                             (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.62,
#                             (0, 80, 255), 1, cv2.LINE_AA)
#                 cv2.putText(disp, f"wajah hilang {face_lost_for:.1f}s "
#                                   f"(timer lanjut)", (12, 52),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
#                             cv2.LINE_AA)
#             else:
#                 cv2.putText(disp, "Wajah tidak terdeteksi", (12, 26),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 2,
#                             cv2.LINE_AA)
#         else:
#             col = (0, 80, 255) if eyes_closed else (0, 255, 120)
#             cv2.putText(disp, f"skor {score_ema:+.2f}", (12, 26),
#                         cv2.FONT_HERSHEY_DUPLEX, 0.62, col, 1, cv2.LINE_AA)
#             st = f"TERTUTUP  {closed_duration:.1f}s" if eyes_closed else "TERBUKA"
#             cv2.putText(disp, st, (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.55, col, 2, cv2.LINE_AA)
#             cv2.putText(disp, f"EAR {vec[0]:.3f}   kedip {blink_count}",
#                         (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
#                         (170, 170, 170), 1, cv2.LINE_AA)

#         # ── Overlay status gyro ──
#         if gyro_ok:
#             gyro_col = (0, 80, 255) if (head_down or recent_jerk) else (0, 255, 120)
#             posture = f"MENUNDUK {head_down_duration:.1f}s" if head_down else "tegak"
#             nod_txt = f"  nod {nod_score_now:+.2f}" if nod_model is not None else ""
#             gyro_txt = (f"gyro: {g['pitch']:.0f}deg {posture}{nod_txt}" +
#                         ("  [ANGGUK]" if recent_jerk else ""))
#         else:
#             gyro_col = (0, 165, 255)
#             gyro_txt = "gyro: TERPUTUS - cek GYRO_IP & WiFi (fallback kamera saja)"
#         cv2.putText(disp, gyro_txt, (12, h - 34),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, gyro_col, 1, cv2.LINE_AA)

#         # Bar skor
#         bx, by, bw_, bh_ = w - 152, 14, 132, 10
#         lo = prof["closed_ceiling"] - 0.6 * prof["gap"]
#         hi = prof["awake_floor"] + 0.9 * prof["gap"]
#         rng = max(hi - lo, 1e-6)
#         cv2.rectangle(disp, (bx, by), (bx + bw_, by + bh_), (70, 70, 70), -1)
#         if score_ema is not None:
#             f = int(bw_ * float(np.clip((score_ema - lo) / rng, 0, 1)))
#             cv2.rectangle(disp, (bx, by), (bx + f, by + bh_),
#                           (0, 80, 255) if eyes_closed else (0, 255, 120), -1)
#         for tv, tc in ((t_close, (0, 200, 255)), (t_open, (255, 200, 0))):
#             tx = bx + int(bw_ * float(np.clip((tv - lo) / rng, 0, 1)))
#             cv2.line(disp, (tx, by - 3), (tx, by + bh_ + 3), tc, 1)
#         cv2.rectangle(disp, (bx, by), (bx + bw_, by + bh_), (140, 140, 140), 1)

#         if eyes_closed:
#             # Target = jalur tercepat yang saat ini memenuhi syarat gyro
#             if recent_jerk:
#                 target = JERK_EYE_DURATION
#             elif gyro_ok and head_down:
#                 target = GYRO_FUSED_EYE_DURATION
#             else:
#                 target = MICROSLEEP_DURATION
#             p = min(closed_duration / target, 1.0)
#             cv2.rectangle(disp, (0, h - 7), (int(w * p), h), (0, 80, 255), -1)

#         if alert_active:
#             ov = disp.copy()
#             cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 200), -1)
#             cv2.addWeighted(ov, 0.30, disp, 0.70, 0, disp)
#             cv2.putText(disp, "! MICROSLEEP !", (w // 2 - 148, h // 2),
#                         cv2.FONT_HERSHEY_DUPLEX, 1.15, (0, 0, 255), 3,
#                         cv2.LINE_AA)
#             cv2.putText(disp, f"{closed_duration:.1f}s  ({alert_reason})",
#                         (w // 2 - 90, h // 2 + 38), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.62, (0, 120, 255), 2, cv2.LINE_AA)

#         fps_n += 1
#         if now - fps_t >= 1.0:
#             fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now
#         cv2.putText(disp, f"{fps:.0f} fps", (w - 60, h - 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1,
#                     cv2.LINE_AA)

#         cv2.imshow("Microsleep Detector", disp)
#         k = cv2.waitKey(1) & 0xFF
#         if k == ord('q'):
#             return "quit"
#         if k == ord('r'):
#             return "recalibrate"
#         if k == ord('e'):
#             show_roi = not show_roi
#         if k == ord('d') and vec is not None:
#             print("[debug] " + "  ".join(
#                 f"{n}={v:.3f}" for n, v in zip(FEATURE_NAMES, vec)) +
#                 f"  skor={score_ema:+.3f}  |  gyro_ok={gyro_ok} "
#                 f"pitch={g['pitch']:.1f} head_down={head_down} "
#                 f"head_dur={head_down_duration:.2f}")


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════

"""
Microsleep Detector MULTI-FITUR + LDA + FUSI GYRO
==================================================
VERSI SATU MIKROKONTROLER (ESP32-S3 WROOM CAM N16R8)

Semua sensor kini berada di SATU board:
  - OV2640      -> streaming JPEG di port 1234
  - GY-521      -> streaming JSON di port 1235
  - Buzzer      -> dikendalikan laptop lewat port 1235

Dua sumber data yang difusi:
  1. KAMERA -> skor LDA dari EAR + fitur tampilan mata
  2. GYRO   -> sudut menunduk (pitch) + angguk terkantuk (LDA gerakan)

  Aturan fusi (3 jalur, mana pun yang lebih dulu terpenuhi):
    A. Kamera SENDIRIAN, mata tertutup >= MICROSLEEP_DURATION (1.5s)
       -> fallback aman kalau data gyro terputus.
    B. Kamera + kepala MENUNDUK bertahan >= 0.8s
       -> pola tertidur perlahan, kepala terkulai.
    C. Kamera + ANGGUKAN terdeteksi, mata tertutup >= 0.5s
       -> pola microsleep klasik: kepala jatuh lalu tersentak.

  Buzzer dikendalikan sepenuhnya dari laptop lewat gyro.set_buzzer().

Install:
    pip install mediapipe opencv-python numpy

TIDAK butuh file pendukung lain — GyroClient sudah disatukan ke file ini.
Model face_landmarker.task diunduh otomatis saat pertama dijalankan.

Jalankan:
    python microsleep_single_esp32s3.py

Kontrol saat DETEKSI:
    q  -> keluar        r  -> kalibrasi ulang
    d  -> debug fitur   e  -> tampil/sembunyikan kotak ROI mata
"""

import os
# Redam log telemetri MediaPipe (pesan "clearcut" yang mengganggu)
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

import socket
import time
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.gyroClient import GyroClient
from src.cameraClient import CameraClient
from src.screenClient import ScreenClient
from src.server import Server
from src.sleepDetectorEngine import SleepDetectorEngine
from src.esp32Listener import resolve_esp32_ip
from src.utility import Utility

def main():

    # DEVICE_IP = "10.187.95.79"    
    # DEVICE_IP = socket.gethostbyname("microsleep.local")
    try:
        DEVICE_IP = resolve_esp32_ip()
    except Exception as e:
        print(f"[!] mDNS Discovery failed: {e}")
        # Fallback to manual entry or static IP if discovery fails
        DEVICE_IP = input("Enter ESP32 IP manually: ")

    BOOT_MELODY_TIME = 1.0     # durasi melodi 'B'
    HAT_ADJUST_TIME  = 10.0    # jeda membenahi posisi topi (sebelum fase 1)
    # PREP_TIME        = 3.0     # jeda membaca instruksi antar fase
    # COUNTDOWN_LEAD   = 2.0     # dari 'C' dikirim s/d beep panjang berbunyi
    COUNTDOWN_TOTAL  = 2.6     # total durasi pola hitung mundur
    END_BEEP_TIME    = 0.8     # durasi beep penanda fase selesai

    # Ambang kepercayaan MediaPipe. Diturunkan karena wajah terdistorsi
    # lensa wide lebih sulit dikenali daripada wajah dari lensa normal.
    DETECT_CONFIDENCE  = 0.15
    PRESENCE_CONFIDENCE = 0.15
    TRACKING_CONFIDENCE = 0.15


    # init all objects
    util = Utility()
    camClient = CameraClient(ip=DEVICE_IP)
    gyClient = GyroClient(ip=DEVICE_IP)
    scrClient = ScreenClient()
    web_server = Server(host="0.0.0.0", port=8000)
    sdengine = SleepDetectorEngine()

    # Init the web server
    web_server.run_in_thread()

    # ensure the model (install the model)
    Utility.ensure_model()

    # ── Detektor utama: mode VIDEO (cepat, pakai pelacakan) ──
    # Ambang diturunkan karena wajah dari lensa wide terdistorsi dan
    # lebih sulit dikenali daripada dari lensa normal.
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=Utility.MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=DETECT_CONFIDENCE,
        min_face_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    detector = vision.FaceLandmarker.create_from_options(opts)

    # ── Detektor cadangan: mode IMAGE (deteksi penuh dari nol) ──
    # Dipakai hanya saat pelacakan VIDEO gagal beberapa frame berturut-turut,
    # supaya sistem bisa mengunci ulang wajah tanpa menunggu pelacakan pulih.
    opts_img = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=Utility.MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=DETECT_CONFIDENCE,
        min_face_presence_confidence=PRESENCE_CONFIDENCE,
    )
    detector_img = vision.FaceLandmarker.create_from_options(opts_img)

    print(f"Menghubungkan ke ESP32-CAM {camClient._ip}:{camClient._port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((camClient._ip, camClient._port))
    print("Terhubung ke kamera.\n")

    print(f"Menghubungkan ke ESP32 gyro {gyClient._ip}:{gyClient._port} ...")
    gyro = gyClient
    gyro.start()

    if not gyro.wait_for_gyro(timeout=10.0, label="Menunggu data gyro"):
        print("      Sistem tetap jalan; gyro akan terus dicoba di latar.")
        print(f"      Pastikan GYRO_IP={GyroClient.GYRO_IP} benar dan ESP32 gyro")
        print("      ada di WiFi/hotspot yang sama.\n")

    ts = [0]

    prof = sdengine.load_profile()
    if prof is not None:
        print(f"Profil MATA ditemukan ({prof.get('waktu', '?')})")
        if input("Pakai profil mata ini? [Y/n] ").strip().lower() == "n":
            prof = None
        print()

    head_prof = sdengine.load_head_profile()
    if head_prof is not None:
        print(f"Profil KEPALA ditemukan ({head_prof.get('waktu', '?')})")
        print(f"  pitch_th={head_prof['pitch_threshold']:.1f}deg", end="")
        if head_prof.get("nod_threshold") is not None:
            print(f"  nod_th={head_prof['nod_threshold']:+.3f}")
        else:
            print("  (deteksi angguk belum terkalibrasi)")
        if input("Pakai profil kepala ini? [Y/n] ").strip().lower() == "n":
            head_prof = None
        print()

    try:

        while True:
            # ═══════════════════════════════════════════
            #  RANGKAIAN KALIBRASI 12 FASE (6 mata + 6 kepala)
            #  Dipandu buzzer sepenuhnya — tidak perlu tekan tombol.
            # ═══════════════════════════════════════════
            if prof is None:
                buzzer_ok = gyro.gyro_ready(gyro)

                if buzzer_ok:
                    # Melodi pembuka + jeda membenahi posisi topi
                    gyro.buzz(Utility.BUZZ_BOOT)
                    print("\n" + "=" * 60)
                    print("  KALIBRASI DIPANDU BUZZER — 12 FASE")
                    print("=" * 60)
                    print("  Melodi pembuka berbunyi.")
                    print(f"  Jeda {HAT_ADJUST_TIME:.0f} detik untuk membenahi posisi topi.")
                    print("  Pola bunyi tiap fase:")
                    print("    beep .. beep .. BEEEP  -> aba-aba MULAI")
                    print("    (hening)               -> sedang merekam")
                    print("    BEEEEEP panjang        -> fase selesai\n")

                    if not scrClient.screen_wait(
                            sock, BOOT_MELODY_TIME + HAT_ADJUST_TIME,
                            "KALIBRASI AKAN DIMULAI",
                            ["Pakai topi dan benahi posisinya sekarang.",
                             "Duduk seperti posisi mengemudi normal.",
                             "",
                             "Fase 1 dimulai otomatis setelah hitungan ini."]):
                        print("Kalibrasi dibatalkan.")
                        break
                    first_prep = Utility.PREP_TIME
                else:
                    print("\n  [!] Buzzer tidak tersedia (gyro belum terhubung).")
                    print("      Kalibrasi tetap jalan, dipandu layar saja.\n")
                    first_prep = HAT_ADJUST_TIME

                data = sdengine.run_calibration(sock, detector, detector_img, ts,
                                       gyro=gyro if buzzer_ok else None,
                                       first_prep=first_prep,
                                       web_server=web_server,
                                       cam_client=camClient,
                                       scr_client=scrClient
                                       )
                if data is None:
                    if buzzer_ok:
                        gyro.buzz(Utility.BUZZ_FAIL)
                    print("Kalibrasi dibatalkan.")
                    break
                prof = sdengine.build_profile(data)
                if prof is None:
                    if buzzer_ok:
                        gyro.buzz(Utility.BUZZ_FAIL)
                        time.sleep(1.2)
                    print("Ulangi kalibrasi.\n")
                    continue
                sdengine.save_profile(prof)

            # ── Kalibrasi KEPALA ──
            if head_prof is None:
                if not gyro.gyro_ready(gyro):
                    # Jangan langsung menyerah — beri waktu reconnect.
                    gyro.wait_for_gyro(gyro, timeout=15.0,
                                  label="Gyro belum siap, menunggu")

                while head_prof is None:
                    if gyro.gyro_ready(gyro):
                        hdata = sdengine.run_head_calibration(sock, gyro=gyro, 
                                                              web_server=web_server,
                                                            cam_client=camClient,
                                                            scr_client=scrClient)
                        if hdata is None:
                            gyro.buzz(Utility.BUZZ_FAIL)
                            print("Kalibrasi kepala dibatalkan. Pakai default.\n")
                            head_prof = sdengine.default_head_profile()
                        else:
                            head_prof = sdengine.compute_head_profile(hdata)
                            sdengine.save_head_profile(head_prof)
                            # ── Seluruh 12 fase selesai ──
                            berhasil = head_prof.get("nod_threshold") is not None
                            gyro.buzz(Utility.BUZZ_SUCCESS if berhasil else Utility.BUZZ_FAIL)
                            if berhasil:
                                print("=" * 60)
                                print("  KALIBRASI 12 FASE SELESAI — SISTEM SIAP")
                                print("=" * 60 + "\n")
                            else:
                                print("  [!] Kalibrasi selesai tapi pemisahan angguk lemah.")
                                print("      Pertimbangkan mengulang fase kepala.\n")
                            time.sleep(1.6)   # biarkan melodi selesai
                        break

                    print("\n" + "!" * 58)
                    print("  GYRO TIDAK TERHUBUNG")
                    print("!" * 58)
                    print(f"  Target: {GyroClient.GYRO_IP}:{GyroClient.GYRO_PORT}")
                    print("  Cek berurutan:")
                    print("    1. ESP32 gyro menyala & Serial Monitor cetak IP?")
                    print("    2. IP di Serial Monitor SAMA dengan GYRO_IP di atas?")
                    print("    3. ESP32 gyro & laptop di hotspot/WiFi yang SAMA?")
                    print("    4. Firmware sudah versi v2 (WiFi.setSleep(false))?")
                    print("\n  [c] coba lagi (tunggu 15 detik)")
                    print("  [s] lewati, jalan dengan kamera saja")
                    pilih = input("  Pilihan [c/s]: ").strip().lower()
                    if pilih == "s":
                        print("Kalibrasi kepala dilewati. Sistem pakai kamera saja.\n")
                        head_prof = sdengine.default_head_profile()
                    else:
                        gyro.wait_for_gyro(gyro, timeout=15.0, label="Mencoba lagi")

            act = sdengine.run_detection(sock, detector, detector_img, ts, prof,
                                gyro, head_prof, web_server=web_server, 
                                scr_client=scrClient, cam_client=camClient)
            if act == "recalibrate":
                prof = None
                head_prof = None
                continue
            break
    finally:
        gyro.set_buzzer(False)
        gyro.stop()
        sock.close()
        cv2.destroyAllWindows()
        detector.close()
        detector_img.close()


if __name__ == "__main__":
    main()