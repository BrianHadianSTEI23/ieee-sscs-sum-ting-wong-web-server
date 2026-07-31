import threading
import socket
import time
import json
from collections import deque
from typing import Final

# ═════════════════════════════════════════════════════════════════
#  GyroClient — sebelumnya file terpisah (gyro_client.py),
#  sekarang disatukan ke sini karena hanya ada 1 mikrokontroler.
#  Isi class TIDAK diubah sama sekali dari versi yang sudah teruji.
# ═════════════════════════════════════════════════════════════════


class GyroClient:

    # ═════════════════════════════════════════════
    #  KONFIGURASI — GYRO + FUSI
    # ═════════════════════════════════════════════
    GYRO_IP   : Final[str] = "192.168.1.5"         # gyro: perangkat yang SAMA dengan kamera (FALLBACK VALUE)
    GYRO_PORT : Final[int] = 1235

    # Data gyro dianggap basi kalau lebih lama dari ini.
    # Dilonggarkan ke 2.5s karena link WiFi hotspot (terutama hotspot HP)
    # sering mengirim data bergerombol dengan jeda 1-2 detik. Nilai ketat
    # seperti 1.0s membuat gyro terus-menerus dianggap "terputus" padahal
    # koneksinya sebenarnya hidup.
    GYRO_MAX_AGE : Final[float] = 2.5

    def __init__(self, ip = GYRO_IP, port = GYRO_PORT, reconnect_interval=2.0, buffer_size=400):
        self._ip = ip
        self._port = port
        self.reconnect_interval = reconnect_interval

        self._sock = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        self._latest = {
            "pitch": 0.0, "roll": 0.0, "rate": 0.0, "accdev": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0, "prate": 0.0,
        }

        # Antrian sampel: tidak ada yang hilang walau data datang bergerombol
        self._samples = deque(maxlen=buffer_size)

        self._connected = False
        self._last_update = 0.0
        self._buzzer_state = None

        self._worst_gap = 0.0
        self._msg_count = 0

    # ─────────────────────────────────────────
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        attempt = 0
        while self._running:
            try:
                self._connect_and_listen()
            except socket.timeout:
                attempt += 1
                print(f"[gyro] GAGAL #{attempt}: timeout menghubungi {self._ip}:{self._port}")
                print("       -> IP kemungkinan salah, atau ESP tidak di WiFi yang sama.")
            except ConnectionRefusedError:
                attempt += 1
                print(f"[gyro] GAGAL #{attempt}: koneksi ditolak {self._ip}:{self._port}")
                print("       -> ESP hidup tapi port salah, atau server belum siap.")
            except OSError as e:
                attempt += 1
                print(f"[gyro] GAGAL #{attempt}: {e}")
                print("       -> Cek laptop & ESP di jaringan/hotspot yang sama.")
            except Exception as e:
                attempt += 1
                print(f"[gyro] koneksi terputus: {e}")

            with self._lock:
                self._connected = False
            if self._running:
                time.sleep(self.reconnect_interval)

    def _connect_and_listen(self):
        print(f"[gyro] Menghubungkan ke {self._ip}:{self._port} ...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((self._ip, self._port))

        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        except OSError:
            pass

        s.settimeout(1.0)
        with self._lock:
            self._sock = s
            self._connected = True
        print("[gyro] Terhubung.")

        if self._buzzer_state is not None:
            self._send_raw(self._buzzer_state)

        buf = b""
        while self._running:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("ESP32 gyro menutup koneksi")

            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._parse_line(line)

    def _parse_line(self, line):
        try:
            obj = json.loads(line.decode("utf-8").strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        try:
            sample = {
                "pitch":  float(obj.get("pitch",  0.0)),
                "roll":   float(obj.get("roll",   0.0)),
                "rate":   float(obj.get("rate",   0.0)),
                "accdev": float(obj.get("accdev", 0.0)),
                "gx":     float(obj.get("gx",     0.0)),
                "gy":     float(obj.get("gy",     0.0)),
                "gz":     float(obj.get("gz",     0.0)),
                "prate":  float(obj.get("prate",  0.0)),
            }
        except (ValueError, TypeError):
            return

        with self._lock:
            now = time.time()
            if self._last_update:
                gap = now - self._last_update
                if gap > self._worst_gap:
                    self._worst_gap = gap
            self._latest = sample
            self._samples.append(sample)
            self._last_update = now
            self._msg_count += 1

    # ─────────────────────────────────────────
    def drain_samples(self):
        """Ambil SEMUA sampel sejak panggilan terakhir, lalu kosongkan."""
        with self._lock:
            out = list(self._samples)
            self._samples.clear()
            return out

    def get_state(self):
        """Nilai terakhir + status koneksi. Untuk tampilan/overlay."""
        with self._lock:
            age = time.time() - self._last_update if self._last_update else None
            st = dict(self._latest)
            st.update({
                "connected": self._connected,
                "age": age,
                "worst_gap": self._worst_gap,
                "msg_count": self._msg_count,
            })
            return st

    def reset_diagnostics(self):
        with self._lock:
            self._worst_gap = 0.0
            self._msg_count = 0

    def buzz(self, cmd: str):
        """
        Kirim perintah POLA buzzer ke ESP32.
        Pola dimainkan di ESP32 (timing presisi), laptop cukup memicu.
          'B' melodi mulai kalibrasi   'C' hitung mundur 3..2..1
          'E' fase selesai             'S' sukses     'F' gagal
        """
        self._send_raw(cmd)

    def set_buzzer(self, on: bool):
        val = "1" if on else "0"
        if self._buzzer_state == val:
            return
        self._buzzer_state = val
        self._send_raw(val)

    def _send_raw(self, val):
        with self._lock:
            sock = self._sock
        if sock:
            try:
                sock.sendall(val.encode("ascii"))
            except Exception as e:
                print(f"[gyro] gagal kirim perintah buzzer: {e}")

    def gyro_ready(self) -> bool:
        st = self.get_state()
        return (st["connected"] and st["age"] is not None
                and st["age"] < self.GYRO_MAX_AGE)

    def wait_for_gyro(self, timeout=15.0, label="Menunggu gyro") -> bool:
        time.sleep(1.0)
        print(f"{label} (maks {timeout:.0f} detik)...")
        t0 = time.time()
        last_dot = 0
        while time.time() - t0 < timeout:
            if self.gyro_ready():
                st = self.get_state()
                print(f"  [OK] Gyro siap. pitch={st['pitch']:.1f}  rate={st['rate']:.1f}\n")
                return True
            if time.time() - last_dot > 1.0:
                print(".", end="", flush=True)
                last_dot = time.time()
            time.sleep(0.2)
        print("\n  [!] Gyro belum siap.")
        return False


# ═════════════════════════════════════════════════════════════════
#  (akhir GyroClient)
# ═════════════════════════════════════════════════════════════════
