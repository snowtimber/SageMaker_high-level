import sagemaker
from sagemaker.estimator import Estimator

# Assume session and role are already set up
session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Step 1: Preprocess the data
# (Insert the preprocessing code here)

# Step 2: Run a training job
estimator = Estimator(
    image_uri='your-training-image-uri',
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://your-bucket/output'
)

estimator.fit({
    'train': 's3://your-bucket/preprocessed/train.csv',
    'validation': 's3://your-bucket/preprocessed/validation.csv'
})

# Step 3: Create a SageMaker model
model = estimator.create_model()

# Step 4: Deploy the model to an inference endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Now you can use the predictor to make inferences
result = predictor.predict(your_input_data)