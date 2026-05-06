#!/usr/bin/env python3
"""Host-side relay. Runs as a systemd service on the host.
Handles Wake-on-LAN and DDNS updates on behalf of the bot container.
Bound to 0.0.0.0 but guarded by _is_allowed() which only accepts private IPs.
"""
import ipaddress
import os
import requests
from http.server import BaseHTTPRequestHandler, HTTPServer
from wakeonlan import send_magic_packet

_ALLOWED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("10.0.0.0/8"),        # private class A
    ipaddress.ip_network("172.16.0.0/12"),     # private class B (includes Docker bridge 172.17.x.x)
    ipaddress.ip_network("192.168.0.0/16"),    # private class C / typical LAN
]


def _is_allowed(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return any(addr in net for net in _ALLOWED_NETWORKS)
    except ValueError:
        return False


def _mangle_ipv6(ip: str) -> str:
    last = ip[-1]
    mangled = format((int(last, 16) + 3) % 16, "x")
    return ip[:-1] + mangled


def _noip_set(ip: str) -> tuple[str, str]:
    resp = requests.get(
        "https://dynupdate.no-ip.com/nic/update",
        params={"hostname": NOIP_HOSTNAME, "myip": ip},
        auth=(NOIP_USERNAME, NOIP_PASSWORD),
        headers={"User-Agent": "JARVISBot/1.0 " + NOIP_USERNAME},
        timeout=10,
    )
    resp.raise_for_status()
    return ip, resp.text.strip()


MAC = os.environ["PC_MAC_ADDRESS"]
BROADCAST = os.environ["BROADCAST_IP"]
NOIP_USERNAME = os.environ["NOIP_USERNAME"]
NOIP_PASSWORD = os.environ["NOIP_PASSWORD"]
NOIP_HOSTNAME = os.environ["NOIP_HOSTNAME"]
PORT = int(os.environ.get("WAKE_RELAY_PORT", "9393"))
BIND = os.environ.get("WAKE_RELAY_BIND", "0.0.0.0")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        client_ip = self.client_address[0]
        if not _is_allowed(client_ip):
            print(f"Blocked request from {client_ip}", flush=True)
            self._respond(403, "forbidden")
            return
        if self.path == "/wake":
            try:
                send_magic_packet(MAC, ip_address=BROADCAST)
                self._respond(200, "ok")
            except Exception as e:
                self._respond(500, str(e))
        elif self.path == "/ddns/update":
            try:
                real_ip = requests.get("http://ip1.dynupdate6.no-ip.com/", timeout=10).text.strip()
                ip, status = _noip_set(real_ip)
                self._respond(200, f"{ip} {status}")
            except Exception as e:
                self._respond(500, str(e))
        elif self.path == "/ddns/block":
            try:
                real_ip = requests.get("http://ip1.dynupdate6.no-ip.com/", timeout=10).text.strip()
                ip, status = _noip_set(_mangle_ipv6(real_ip))
                self._respond(200, f"{ip} {status}")
            except Exception as e:
                self._respond(500, str(e))
        else:
            self._respond(404, "not found")

    def _respond(self, code: int, body: str):
        encoded = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)


if __name__ == "__main__":
    server = HTTPServer((BIND, PORT), Handler)
    print(f"Wake relay listening on {BIND}:{PORT}", flush=True)
    server.serve_forever()
