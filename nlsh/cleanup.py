import os
from pathlib import Path

#root folder για καθάρισμα
root = Path(".")

#διαγράφει όλα τα .csv
for csv_file in root.rglob("*.csv"):
    print(f"Deleting {csv_file}")
    csv_file.unlink()

#διαγράφει όλα τα kahip_partitions.txt
for kahip_file in root.rglob("kahip_partitions.txt"):
    print(f"Deleting {kahip_file}")
    kahip_file.unlink()

print("Cleanup completed.")
