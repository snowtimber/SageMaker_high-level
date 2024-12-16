# AWS SageMaker End-to-End Machine Learning Workflow

This project demonstrates a complete machine learning workflow using AWS SageMaker, from data preprocessing to model deployment and inference. It includes data processing, model training using XGBoost, optional hyperparameter tuning, deployment to a SageMaker endpoint, and making predictions.

## Prerequisites

- An AWS account with SageMaker access
- AWS CLI configured with appropriate permissions
- Python 3.7+
- AWS SDK for Python (Boto3)
- SageMaker Python SDK

## Project Structure

- `SageMaker_HighLevel.py`: Main script orchestrating the entire SageMaker workflow
- `processing-script.py`: Data preprocessing script
- `xgboost-script.py`: XGBoost training script (to be created by user)
- `raw_data.csv`: Your input data (to be provided by user)

## Setup

1. Clone this repository to your local machine or SageMaker notebook instance.
2. Upload your `raw_data.csv` to an S3 bucket.
3. Create `xgboost-script.py` with your XGBoost training code.
4. Update the `bucket` variable in `SageMaker_HighLevel.py` with your S3 bucket name.

## Workflow Steps

### 1. Data Processing

The `processing-script.py` handles data preprocessing:
- Reads the input data
- Handles missing values
- Encodes categorical variables
- Splits data into training and validation sets

### 2. Model Training

Uses SageMaker's XGBoost algorithm to train a model:
- Configures an XGBoost estimator with initial hyperparameters
- Trains the model on the processed data

### 3. Hyperparameter Tuning (Optional)

Optionally uses SageMaker's Hyperparameter Tuning to find the best hyperparameters:
- Defines hyperparameter ranges for XGBoost
- Runs multiple training jobs with different hyperparameter combinations
- Selects the best performing model based on the validation AUC

### 4. Model Creation

Creates a SageMaker model object using either:
- The model from the initial training, or
- The best model from hyperparameter tuning (if performed)

### 5. Model Deployment

Deploys the trained model to a SageMaker endpoint for real-time inference.

### 6. Inference

Demonstrates how to use the deployed endpoint to make predictions on new data.

## Usage

1. Ensure all prerequisites are met and setup is complete.
2. Run the main script:

   ```
   python SageMaker_HighLevel.py
   ```

3. When prompted, choose whether to perform hyperparameter tuning.
4. Monitor the progress in the AWS SageMaker console.
5. Once complete, the script will output the endpoint name and demonstrate making a prediction.

## Making Predictions

After deploying the model, the script demonstrates how to use the SageMaker endpoint to make predictions:

1. It prepares sample data (which you should replace with your actual test data).
2. It sends this data to the deployed endpoint using the SageMaker runtime client.
3. It receives and displays the prediction results.

You can adapt this code to make predictions on new data in your production environment.

## Customization

- Modify `processing-script.py` to suit your specific data preprocessing needs.
- Adjust hyperparameters and their ranges in `SageMaker_HighLevel.py` for your use case.
- Update instance types and counts based on your dataset size and complexity.
- Replace the sample data generation in Step 6 with code to load and preprocess your real test data.

## Cleaning Up

The script includes a step to delete the SageMaker endpoint after making a prediction. This is to avoid unnecessary charges. In a production setting, you might want to keep the endpoint running and manage its lifecycle separately.

To manually delete resources:

```python
import boto3

sagemaker = boto3.client('sagemaker')
sagemaker.delete_endpoint(EndpointName='your-endpoint-name')
```

## Troubleshooting

- Check CloudWatch logs for detailed error messages and logs.
- Ensure your IAM role has necessary permissions for S3, SageMaker, and other used services.
- Verify that your data format matches the expected input for each step.
- If encountering issues with the endpoint invocation, check that your input data matches the expected format and features used during training.

## Best Practices

- Version your data and models.
- Use SageMaker Experiments to track your ML iterations.
- Implement proper error handling and logging for production use.
- Regularly monitor your deployed models for performance degradation.
- Consider using SageMaker Model Monitor for production deployments.

## Further Resources

- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.