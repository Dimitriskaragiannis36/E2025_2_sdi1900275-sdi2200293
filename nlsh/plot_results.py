import os #για διαχείριση αρχείων και καταλόγων
import re #για κανονικές εκφράσεις
import pandas as pd #για διαχείριση δεδομένων
import matplotlib.pyplot as plt #για γραφήματα
import seaborn as sns #για βελτιωμένα γραφήματα
from matplotlib.backends.backend_pdf import PdfPages #για αποθήκευση πολλαπλών γραφημάτων σε PDF
from openpyxl import load_workbook #για εργασία με αρχεία Excel

#αρχεία αποτελεσμάτων
MNIST_FILE = "output/mnist/final_output.txt"
SIFT_FILE = "output/sift/final_output.txt"

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True) #δημιουργία καταλόγου για αποθήκευση γραφημάτων

#πρότυπα για εξαγωγή μετρικών
metrics_pattern = {
    "Average AF": r"Average AF:\s*([0-9.eE+-]+)",
    "Recall@N": r"Recall@N:\s*([0-9.eE+-]+)",
    "QPS": r"QPS:\s*([0-9.eE+-]+)",
    "tApproximateAverage": r"tApproximateAverage:\s*([0-9.eE+-]+)",
    "tTrueAverage": r"tTrueAverage:\s*([0-9.eE+-]+)"
}

#συνάρτηση για ανάλυση αρχείου αποτελεσμάτων
def parse_results_file(filepath, dataset_name):
    if not os.path.exists(filepath): #έλεγχος ύπαρξης αρχείου
        print(f"WARNING: File not found → {filepath}")
        return None

    with open(filepath, "r") as f: #ανάγνωση περιεχομένου αρχείου
        text = f.read() #όλο το κείμενο

    data = {"Dataset": dataset_name, "Method": "NLSH"} #αρχικοποίηση λεξικού με όνομα συνόλου δεδομένων και μέθοδο

    for key, pattern in metrics_pattern.items(): #εξαγωγή μετρικών
        match = re.search(pattern, text) #αναζήτηση με κανονική έκφραση
        data[key] = float(match.group(1)) if match else None #αποθήκευση τιμής ή None αν δεν βρέθηκε

    return data


#ανάγνωση και ανάλυση αρχείων αποτελεσμάτων
records = []

mnist_data = parse_results_file(MNIST_FILE, "MNIST")
if mnist_data: records.append(mnist_data)

sift_data  = parse_results_file(SIFT_FILE, "SIFT")
if sift_data:  records.append(sift_data)

if not records:
    print("ERROR: No final_output.txt files found.")
    exit(1)

df = pd.DataFrame(records) #δημιουργία DataFrame από τα δεδομένα

print("\nΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ:\n")
print(df.to_string(index=False)) #εκτύπωση πίνακα αποτελεσμάτων

#αποθήκευση πινάκων αποτελεσμάτων
df.to_csv(f"{PLOTS_DIR}/results.csv", index=False)

try:

    excel_path = f"{PLOTS_DIR}/results.xlsx" #διαδρομή αποθήκευσης Excel αρχείου

    #γράψιμο Excel αρχείου όπως πριν
    df.to_excel(excel_path, index=False)

    #ανοίγουμε το υπάρχον Excel
    wb = load_workbook(excel_path)
    ws = wb.active

    #format για 4 δεκαδικά
    fmt = "0.0000"

    #για κάθε κελί στους αριθμητικούς πίνακες του DataFrame
    for row in ws.iter_rows(min_row=2):   #ξεκινάμε από τη 2η γραμμή (δεδομένα)
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = fmt #εφαρμογή format

    wb.save(excel_path) #αποθήκευση αλλαγών

except:
    print("WARNING: openpyxl not installed → Excel file not created.")

#διαγγράμματα
sns.set(style="whitegrid", font_scale=1.15)
plots = [] #λίστα για αποθήκευση γραφημάτων

#συνάρτηση για δημιουργία γραφήματος
def plot_metric(metric, title, ylabel):
    plt.figure(figsize=(7,5)) #μέγεθος γραφήματος
    ax = sns.barplot(data=df, x="Dataset", y=metric, hue="Method", palette="Set2") #ραβδογράφημα
    plt.title(title) #τίτλος γραφήματος
    plt.ylabel(ylabel) #ετικέτα άξονα y
    plt.xlabel("Dataset") #ετικέτα άξονα x

    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f") #ετικέτες στις ράβδους

    plt.tight_layout() #βελτιστοποίηση διάταξης
    filename = metric.replace("@", "_at_").replace(" ", "_") #δημιουργία ονόματος αρχείου
    path = f"{PLOTS_DIR}/{filename}.png" #πλήρης διαδρομή αποθήκευσης
    plt.savefig(path, dpi=150) #αποθήκευση γραφήματος
    plots.append(plt.gcf()) #προσθήκη γραφήματος στη λίστα

#τα 5 γραφήματα
plot_metric("Average AF", "Average Approximation Factor", "AF")
plot_metric("Recall@N", "Recall@N", "Recall")
plot_metric("QPS", "Queries Per Second", "QPS")
plot_metric("tApproximateAverage", "Average Approx Query Time (ms)", "ms")
plot_metric("tTrueAverage", "Average True Query Time (ms)", "ms")

#scatter plot για trade-off
plt.figure(figsize=(7,5)) #μέγεθος γραφήματος
sns.scatterplot(data=df, x="QPS", y="Average AF", hue="Dataset", s=150) #scatter plot
plt.title("Speed vs Approximation Quality") #τίτλος γραφήματος
plt.xlabel("QPS") #ετικέτα άξονα x
plt.tight_layout() #βελτιστοποίηση διάταξης
plt.savefig(f"{PLOTS_DIR}/tradeoff.png", dpi=150) #αποθήκευση γραφήματος
plots.append(plt.gcf()) #προσθήκη γραφήματος στη λίστα

#αποθήκευση όλων των γραφημάτων σε ένα PDF
with PdfPages(f"{PLOTS_DIR}/summary_report.pdf") as pdf:
    for fig in plots: #εγγραφή κάθε γραφήματος στο PDF
        pdf.savefig(fig) #αποθήκευση γραφήματος στο PDF

print(f"\n ΟΛΑ ΤΑ PLOTS & ΠΙΝΑΚΕΣ ΑΠΟΘΗΚΕΥΤΗΚΑΝ ΣΤΟ: {PLOTS_DIR}/")
