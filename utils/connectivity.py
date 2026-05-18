import socket
import logging

log = logging.getLogger(__name__)

def check_internet(host="8.8.8.8", port=53, timeout=2):
    """
    Check if the internet is available by attempting to connect to a reliable DNS server.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False
