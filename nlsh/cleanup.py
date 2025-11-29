import shutil #για διαγραφή φακέλων με περιεχόμενά τους
from pathlib import Path #για διαχείριση διαδρομών αρχείων και φακέλων

#root folder για καθάρισμα
root = Path(".")

#διαγράφει όλα τα .csv
for csv_file in root.rglob("*.csv"): #rglob ψάχνει αναδρομικά σε όλους τους υποφακέλους
    print(f"Deleting {csv_file}")
    csv_file.unlink() #διαγράφει το αρχείο

#διαγράφει όλα τα output subfolders με περιεχόμενά τους
for output_dir in root.rglob("output"): #rglob ψάχνει αναδρομικά σε όλους τους υποφακέλους
    if output_dir.is_dir(): #ελέγχει αν είναι φάκελος
        print(f"Deleting folder {output_dir}")
        shutil.rmtree(output_dir) #διαγράφει τον φάκελο και όλα τα περιεχόμενά του

#διαγράφει όλα τα nlsh_index subfolders με περιεχόμενά τους
for nlsh_index_dir in root.rglob("nlsh_index"): #rglob ψάχνει αναδρομικά σε όλους τους υποφακέλους
    if nlsh_index_dir.is_dir(): #ελέγχει αν είναι φάκελος
        print(f"Deleting folder {nlsh_index_dir}")
        shutil.rmtree(nlsh_index_dir) #διαγράφει τον φάκελο και όλα τα περιεχόμενά του

#διαγράφει όλα τα plots subfolders με περιεχόμενά τους
for plots_dir in root.rglob("plots"): #rglob ψάχνει αναδρομικά σε όλους τους υποφακέλους
    if plots_dir.is_dir(): #ελέγχει αν είναι φάκελος
        print(f"Deleting folder {plots_dir}") 
        shutil.rmtree(plots_dir) #διαγράφει τον φάκελο και όλα τα περιεχόμενά του

print("Cleanup completed.")
