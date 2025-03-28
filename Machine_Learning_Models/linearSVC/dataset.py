from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import  train_test_split
import pandas as pd

dataset_df = pd.read_csv("../../Dataset/train_dataset.csv")

features = dataset_df.drop(columns=['slice Type'])
label = dataset_df['slice Type'] - 1

mMScaler = MinMaxScaler(feature_range=(0,1))
scaledFeatures = mMScaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(scaledFeatures, label, test_size=0.2, random_state=42, stratify=label)