#!/usr/bin/env python3
"""Host-side Wake-on-LAN relay. Runs as a systemd service on the host.
The bot container calls http://172.17.0.1:<PORT>/wake instead of sending
a broadcast directly (which bridge networking blocks).
"""
import ipaddress
import os
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

MAC = os.environ["PC_MAC_ADDRESS"]
BROADCAST = os.environ["BROADCAST_IP"]
PORT = int(os.environ.get("WAKE_RELAY_PORT", "9393"))
BIND = os.environ.get("WAKE_RELAY_BIND", "172.17.0.1")


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
