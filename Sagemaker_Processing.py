import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import boto3
import io

# Step 1: Preprocess the data

# Load the data (assuming it's in a CSV file in S3)
s3 = boto3.client('s3')
bucket = 'your-bucket-name'
key = 'path/to/your/data.csv'

obj = s3.get_object(Bucket=bucket, Key=key)
data = pd.read_csv(io.BytesIO(obj['Body'].read()))

# Assume 'target' is the name of your target variable column
# Move the target variable to the first column
cols = list(data.columns)
cols.insert(0, cols.pop(cols.index('target')))
data = data.reindex(columns=cols)

# Split the data into features and target
X = data.iloc[:, 1:]  # All columns except the first
y = data.iloc[:, 0]   # First column

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Combine the scaled features with the target variable
train_data = pd.concat([pd.DataFrame(y_train).reset_index(drop=True), 
                        pd.DataFrame(X_train_scaled)], axis=1)
val_data = pd.concat([pd.DataFrame(y_val).reset_index(drop=True), 
                      pd.DataFrame(X_val_scaled)], axis=1)

# Save the preprocessed data back to S3
train_csv_buffer = io.StringIO()
val_csv_buffer = io.StringIO()

train_data.to_csv(train_csv_buffer, index=False, header=False)
val_data.to_csv(val_csv_buffer, index=False, header=False)

s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'preprocessed/train.csv').put(Body=train_csv_buffer.getvalue())
s3_resource.Object(bucket, 'preprocessed/validation.csv').put(Body=val_csv_buffer.getvalue())

print("Preprocessing complete. Data saved to S3.")