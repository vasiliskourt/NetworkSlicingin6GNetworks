import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

dataset_df = pd.read_csv("train_dataset.csv")

# Εμφάνιση περιγραφής των στηλών του Dataset 
print("\n-> Description of Dataset Columns:\n")
with open("description.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)
print("\n============================================================================================")

# Εμφάνιση πληροφοριών του Dataset
print("\n-> Dataset:\n")
print(dataset_df.head(10))
print("\n============================================================================================")

print("\n-> Dataset Column Types:\n")
print(dataset_df.dtypes)
print("\n============================================")

print("\n-> Dataset Labels (Slice Types):\n")
print(dataset_df['slice Type'].value_counts())
print("\n============================================\n")

# Δημιουργία του heatmap συσχέτισης
features = dataset_df.drop(columns=['slice Type'])
print("\n-> Generate feature correlation heatmap")
correlation = features.corr()
print("\n============================================\n")

plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Dataset_plots/correlation_heatmap.png")
plt.close()

# Δημιουργία του heatmap ισχυρών συσχετίσεων  
filtered_corr = correlation.mask((correlation < 0.5) & (correlation > -0.5))
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("Dataset_plots/correlation_heatmap_0_5.png")
plt.close()

# Αποθήκευση πληροφοριών του Dataset
with open("dataset_report.txt", "w") as file:
    file.write("--------- Dataset Report---------\n")
    file.write("\n-> Dataset:\n")
    file.write(dataset_df.head(10).to_string())
    file.write("\n\n-> Dataset Column Types:\n")
    file.write(dataset_df.dtypes.to_string())
    file.write("\n\n-> Dataset Labels (Slice Types):\n")
    file.write(dataset_df['slice Type'].value_counts().to_string())

print("Report saved:\n -> Dataset/\n")
print("Plots saved:\n -> Dataset/Dataset_plots\n")
print("================================\n")