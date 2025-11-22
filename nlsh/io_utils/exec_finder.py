import os
import shutil

def autodetect_exec():
    """
    Βρίσκει το search executable αυτόματα.
    """
    candidates = [
        "./bin/search",
        "bin/search",
        "../bin/search",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    home = os.path.expanduser("~")
    for root, _, files in os.walk(home):
        if "search" in files:
            return os.path.join(root, "search")

    found = shutil.which("search")
    if found:
        return found

    return None
