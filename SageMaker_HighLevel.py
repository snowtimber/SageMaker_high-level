import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoost
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter, CategoricalParameter
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.predictor import Predictor

def main():
    # Set up the SageMaker session and role
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    # Define S3 bucket and prefixes
    bucket = 'your-bucket-name'
    input_prefix = 'raw-data'
    output_prefix = 'processed-data'
    model_prefix = 'model-output'

    # Step 1: Data Processing
    #--------------------------
    print("Step 1: Data Processing")

    script_processor = ScriptProcessor(
        role=role,
        image_uri='your-processing-image-uri',
        command=['python3'],
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=30
    )

    print("Starting the processing job...")
    script_processor.run(
        code=f's3://{bucket}/processing-script.py',
        inputs=[
            ProcessingInput(
                source=f's3://{bucket}/{input_prefix}',
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='train_data',
                source='/opt/ml/processing/output/train',
                destination=f's3://{bucket}/{output_prefix}/train'
            ),
            ProcessingOutput(
                output_name='test_data',
                source='/opt/ml/processing/output/test',
                destination=f's3://{bucket}/{output_prefix}/test'
            )
        ],
        arguments=[
            '--input-data', '/opt/ml/processing/input',
            '--output-train', '/opt/ml/processing/output/train',
            '--output-test', '/opt/ml/processing/output/test'
        ]
    )

    script_processor.jobs[-1].wait()
    print("Processing job completed.")

    # Step 2: Model Training
    #--------------------------
    print("\nStep 2: Model Training")

    # Define the XGBoost estimator
    xgb = XGBoost(
        entry_point='xgboost-script.py',  # Your XGBoost training script
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        framework_version='1.5-1',
        output_path=f's3://{bucket}/{model_prefix}/',
        hyperparameters={
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'num_round': 100,
            'max_depth': 5,
            'eta': 0.2,
            'gamma': 4,
            'min_child_weight': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    )

    # Define the input data for training
    train_data = TrainingInput(
        s3_data=f's3://{bucket}/{output_prefix}/train',
        content_type='csv'
    )
    
    test_data = TrainingInput(
        s3_data=f's3://{bucket}/{output_prefix}/test',
        content_type='csv'
    )

    # Start the training job
    print("Starting the training job...")
    xgb.fit({'train': train_data, 'validation': test_data})
    print("Training job completed.")

    # Step 3: Hyperparameter Tuning (Optional)
    #-----------------------------------------
    print("\nStep 3: Hyperparameter Tuning (Optional)")

    # Ask user if they want to perform hyperparameter tuning
    perform_hpo = input("Do you want to perform hyperparameter tuning? (yes/no): ").lower().strip() == 'yes'

    if perform_hpo:
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            'max_depth': IntegerParameter(3, 10),
            'eta': ContinuousParameter(0.01, 0.3),
            'gamma': ContinuousParameter(0, 5),
            'min_child_weight': ContinuousParameter(1, 10),
            'subsample': ContinuousParameter(0.5, 1.0),
            'colsample_bytree': ContinuousParameter(0.5, 1.0)
        }

        # Create the tuner
        tuner = HyperparameterTuner(
            xgb,
            objective_metric_name='validation:auc',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=10,
            max_parallel_jobs=2,
            strategy='Bayesian',
            objective_type='Maximize'
        )

        # Start the hyperparameter tuning job
        print("Starting the hyperparameter tuning job...")
        tuner.fit({'train': train_data, 'validation': test_data})
        print("Hyperparameter tuning job completed.")

        # Get the best training job
        best_training_job = tuner.best_training_job()
        print(f"Best training job: {best_training_job}")

        # Update xgb with the best model
        xgb = XGBoost.attach(best_training_job)

    # Step 4: Create the Model
    #--------------------------
    print("\nStep 4: Create the Model")

    # Create a model object
    model = Model(
        image_uri=xgb.image_uri,
        model_data=xgb.model_data,
        role=role,
        predictor_cls=Predictor
    )

    print("Model object created.")

    # Step 5: Deploy to Endpoint
    #--------------------------
    print("\nStep 5: Deploy to Endpoint")

    # Deploy the model to an endpoint
    print("Deploying the model to an endpoint...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium'
    )

    print(f"Model deployed. Endpoint name: {predictor.endpoint_name}")

    # Step 6: Invoke the Model Endpoint
    #----------------------------------
    print("\nStep 6: Invoke the Model Endpoint")

    # Create a SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime')

    # Prepare sample data for prediction
    # Replace this with actual feature values from your dataset
    sample_data = np.random.rand(1, 10).tolist()  # Assuming 10 features
    payload = ','.join(map(str, sample_data[0]))

    print("Sample data for prediction:", payload)

    # Make a prediction
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=predictor.endpoint_name,
        ContentType='text/csv',
        Body=payload
    )

    # Parse and print the response
    result = response['Body'].read().decode('ascii')
    print("Prediction result:", result)

    # Clean up (optional)
    print("\nCleaning up...")
    predictor.delete_endpoint()
    print("Endpoint deleted.")

if __name__ == "__main__":
    main()