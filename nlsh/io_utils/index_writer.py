import json #για την αποθήκευση μεταδεδομένων
from typing import Dict, List #για τύπους δεδομένων

#εγγραφή αρχείου κατατμήσεων
def write_partitions_file(blocks: List[int], path: str):
    with open(path, "w") as f: #άνοιγμα αρχείου για εγγραφή
        for node_id, part in enumerate(blocks): #περνάμε από κάθε κόμβο και την αντίστοιχη κατάτμησή του
            f.write(f"{node_id},{part}\n") #εγγραφή κάθε κόμβου και της αντίστοιχης κατάτμησής του

#εγγραφή αντιστραμμένου αρχείου CSV
def write_inverted_csv(parts_map: Dict[int, List[int]], path: str):
    with open(path, "w") as f: #άνοιγμα αρχείου για εγγραφή
        for part in sorted(parts_map.keys()): #ταξινόμηση και περνάμε από κάθε κατάτμηση
            nodes = parts_map[part] #λήψη των κόμβων για την τρέχουσα κατάτμηση
            f.write(",".join([str(part)] + [str(n) for n in nodes]) + "\n") #εγγραφή της κατάτμησης και των κόμβων της

#εγγραφή μεταδεδομένων σε αρχείο JSON
def write_meta(meta: Dict, path: str):
    with open(path, "w") as f: #άνοιγμα αρχείου για εγγραφή
        json.dump(meta, f, indent=2) #εγγραφή των μεταδεδομένων σε μορφή JSON με εσοχές για καλύτερη αναγνωσιμότητα
