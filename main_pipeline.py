from src.preprocessing import load_and_clean_data, perform_eda_and_save_plots, feature_engineering, scale_data
from src.training import train_model_with_tuning, save_artifacts
from src.evaluation import evaluate_model
from sklearn.model_selection import train_test_split
import joblib

# Configuration
DATA_PATH = 'data/creditcard.csv'
EDA_OUTPUT = 'outputs/plots_eda'
RESULTS_OUTPUT = 'outputs/plots_results'
MODEL_OUTPUT = 'outputs/models'

def main():
    print("--- Starting Credit Card Fraud Detection Pipeline ---")
    
    # 1. Load Data
    print("1. Loading Data...")
    df = load_and_clean_data(DATA_PATH)
    
    # 2. EDA (Task 1)
    print("2. Performing Exploratory Data Analysis...")
    perform_eda_and_save_plots(df, EDA_OUTPUT)
    
    # 3. Feature Engineering (Task 3)
    print("3. Feature Engineering...")
    df = feature_engineering(df)
    
    # Separate Features and Target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # 4. Train-Test Split (Crucial before scaling/resampling)
    print("4. Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Scaling (Task 2)
    print("5. Scaling Data...")
    # Note: We fit scaler ONLY on training data, then transform test data
    X_train, scaler = scale_data(X_train)
    X_test['Amount'] = scaler.transform(X_test['Amount'].values.reshape(-1, 1))
    
    # 6. Training with SMOTE & Tuning (Task 4, 6, 7)
    print("6. Training Model with SMOTE and Hyperparameter Tuning...")
    final_pipeline = train_model_with_tuning(X_train, y_train)
    
    # 7. Evaluation & Threshold Tuning (Task 5, 8)
    print("7. Evaluating Model...")
    best_threshold = evaluate_model(final_pipeline, X_test, y_test, RESULTS_OUTPUT)
    
    # 8. Saving Model
    print("8. Saving Artifacts...")
    save_artifacts(final_pipeline, scaler, MODEL_OUTPUT)
    
    # Save the best threshold to a simple file for the app to read
    with open(f"{MODEL_OUTPUT}/best_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    print("--- Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    main()