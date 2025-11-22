import os
import shutil
from pathlib import Path

#root folder για καθάρισμα
root = Path(".")

#διαγράφει όλα τα .csv
for csv_file in root.rglob("*.csv"):
    print(f"Deleting {csv_file}")
    csv_file.unlink()

#διαγράφει όλα τα kahip subfolders με περιεχόμενά τους
for kahip_dir in root.rglob("kahip"):
    if kahip_dir.is_dir():
        print(f"Deleting folder {kahip_dir}")
        shutil.rmtree(kahip_dir)

print("Cleanup completed.")
