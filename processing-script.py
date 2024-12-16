import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def main(args):
    print("Reading input data...")
    input_data_path = os.path.join(args.input_data, "raw_data.csv")
    df = pd.read_csv(input_data_path)
    
    print("Preprocessing the data...")
    # Assume the target variable is in the first column
    target_column = df.columns[0]
    feature_columns = df.columns[1:]

    # Separate features and target
    X = df[feature_columns]
    y = df[target_column]

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    # Combine features and target
    processed_data = pd.concat([y, X], axis=1)

    # Split the data into train and validation sets
    train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)

    print("Saving processed data...")
    # Save training data
    train_output_path = os.path.join(args.output_train, "train.csv")
    train_data.to_csv(train_output_path, index=False, header=False)

    # Save validation data
    val_output_path = os.path.join(args.output_test, "validation.csv")
    val_data.to_csv(val_output_path, index=False, header=False)

    print("Processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-train", type=str, required=True)
    parser.add_argument("--output-test", type=str, required=True)
    args = parser.parse_args()
    main(args)