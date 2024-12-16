import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval_metric', type=str, default='auc')

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args = parser.parse_args()
    return args

def load_data(path):
    # Assumes the first column is the target and the rest are features
    data = pd.read_csv(path, header=None)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    return xgb.DMatrix(X, label=y)

def train(args):
    # Load data
    train_data = load_data(os.path.join(args.train, 'train.csv'))
    validation_data = load_data(os.path.join(args.validation, 'validation.csv'))

    # Set up parameters for XGBoost
    params = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'objective': args.objective,
        'eval_metric': args.eval_metric
    }

    # Train the model
    watchlist = [(train_data, 'train'), (validation_data, 'validation')]
    model = xgb.train(params, train_data, args.num_round, evals=watchlist, early_stopping_rounds=10)

    # Save the model
    model_location = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_location)

    # Calculate and print final AUC
    predictions = model.predict(validation_data)
    auc = roc_auc_score(validation_data.get_label(), predictions)
    print(f"Final Validation AUC: {auc}")

if __name__ == '__main__':
    args = parse_args()
    train(args)