import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

#αρχεία αποτελεσμάτων
MNIST_FILE = "output/mnist/final_output.txt"
SIFT_FILE = "output/sift/final_output.txt"

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    if not os.path.exists(filepath):
        print(f"WARNING: File not found → {filepath}")
        return None

    with open(filepath, "r") as f:
        text = f.read()

    data = {"Dataset": dataset_name, "Method": "NLSH"}

    for key, pattern in metrics_pattern.items():
        match = re.search(pattern, text)
        data[key] = float(match.group(1)) if match else None

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

df = pd.DataFrame(records)

print("\nΠΙΝΑΚΑΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ:\n")
print(df.to_string(index=False))

#αποθήκευση πινάκων αποτελεσμάτων
df.to_csv(f"{PLOTS_DIR}/results.csv", index=False)

try:
    df.to_excel(f"{PLOTS_DIR}/results.xlsx", index=False)
except:
    print("WARNING: openpyxl not installed → Excel file not created.")

#διαγγράμματα
sns.set(style="whitegrid", font_scale=1.15)
plots = []

def plot_metric(metric, title, ylabel):
    plt.figure(figsize=(7,5))
    ax = sns.barplot(data=df, x="Dataset", y=metric, hue="Method", palette="Set2")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Dataset")

    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.3f")

    plt.tight_layout()
    filename = metric.replace("@", "_at_").replace(" ", "_")
    path = f"{PLOTS_DIR}/{filename}.png"
    plt.savefig(path, dpi=150)
    plots.append(plt.gcf())

#τα 5 γραφήματα
plot_metric("Average AF", "Average Approximation Factor", "AF")
plot_metric("Recall@N", "Recall@N", "Recall")
plot_metric("QPS", "Queries Per Second", "QPS")
plot_metric("tApproximateAverage", "Average Approx Query Time (ms)", "ms")
plot_metric("tTrueAverage", "Average True Query Time (ms)", "ms")

#scatter plot για trade-off
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="QPS", y="Average AF", hue="Dataset", s=150)
plt.title("Speed vs Approximation Quality")
plt.xlabel("QPS")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/tradeoff.png", dpi=150)
plots.append(plt.gcf())

#αποθήκευση όλων των γραφημάτων σε ένα PDF
with PdfPages(f"{PLOTS_DIR}/summary_report.pdf") as pdf:
    for fig in plots:
        pdf.savefig(fig)

print(f"\n ΟΛΑ ΤΑ PLOTS & ΠΙΝΑΚΕΣ ΑΠΟΘΗΚΕΥΤΗΚΑΝ ΣΤΟ: {PLOTS_DIR}/")
