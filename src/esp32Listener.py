import socket
import time
from zeroconf import ServiceBrowser, Zeroconf

class ESP32Listener:
    def __init__(self):
        self.found_ip = None

    def remove_service(self, zeroconf, type, name):
        pass

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and info.addresses:
            self.found_ip = socket.inet_ntoa(info.addresses[0])

def resolve_esp32_ip(service_type: str = "_microsleep-cam._tcp.local.", timeout: float = 10.0) -> str:
    """Uses pure Python mDNS discovery to locate the ESP32 IP address."""
    print("[*] Searching for ESP32 on local network via mDNS...")
    zeroconf = Zeroconf()
    listener = ESP32Listener()
    browser = ServiceBrowser(zeroconf, service_type, listener)

    start_time = time.time()
    while listener.found_ip is None and (time.time() - start_time) < timeout:
        time.sleep(0.2)

    zeroconf.close()
    
    if listener.found_ip:
        print(f"[+] Discovered ESP32 IP: {listener.found_ip}")
        return listener.found_ip
    else:
        raise RuntimeError("Failed to resolve ESP32 IP via mDNS. Check Wi-Fi connection or router isolation.")
