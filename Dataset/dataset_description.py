import pandas as pd

pd.set_option('display.max_columns', None)

dataset_df = pd.read_csv("train_dataset.csv")

print("\n-> Description of Dataset Columns:\n")
with open("description.txt", "r", encoding="utf-8") as file:
    content = file.read()
    print(content)
print("\n============================================================================================")

print("\n-> Dataset:\n")
print(dataset_df.head(10))
print("\n============================================================================================")

print("\n-> Dataset Column Types:\n")
print(dataset_df.dtypes)
print("\n============================================")


print("\n-> Dataset Labels (Slice Types):\n")
print(dataset_df['slice Type'].value_counts())
print("\n============================================\n")

