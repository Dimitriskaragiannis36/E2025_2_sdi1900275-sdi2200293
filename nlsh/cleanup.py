import os
import shutil
from pathlib import Path

#root folder για καθάρισμα
root = Path(".")

#διαγράφει όλα τα .csv
for csv_file in root.rglob("*.csv"):
    print(f"Deleting {csv_file}")
    csv_file.unlink()

#διαγράφει όλα τα output*.txt αρχεία
for output_txt in root.rglob("output*.txt"):
    print(f"Deleting {output_txt}")
    output_txt.unlink()

#διαγράφει όλα τα nlsh_index subfolders με περιεχόμενά τους
for nlsh_index_dir in root.rglob("nlsh_index"):
    if nlsh_index_dir.is_dir():
        print(f"Deleting folder {nlsh_index_dir}")
        shutil.rmtree(nlsh_index_dir)

print("Cleanup completed.")
