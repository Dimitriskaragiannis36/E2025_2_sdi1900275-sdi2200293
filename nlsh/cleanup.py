import os
import shutil
from pathlib import Path

#root folder για καθάρισμα
root = Path(".")

#διαγράφει όλα τα .csv
for csv_file in root.rglob("*.csv"):
    print(f"Deleting {csv_file}")
    csv_file.unlink()

#διαγράφει όλα τα output subfolders με περιεχόμενά τους
for output_dir in root.rglob("output"):
    if output_dir.is_dir():
        print(f"Deleting folder {output_dir}")
        shutil.rmtree(output_dir)

#διαγράφει όλα τα nlsh_index subfolders με περιεχόμενά τους
for nlsh_index_dir in root.rglob("nlsh_index"):
    if nlsh_index_dir.is_dir():
        print(f"Deleting folder {nlsh_index_dir}")
        shutil.rmtree(nlsh_index_dir)

print("Cleanup completed.")
