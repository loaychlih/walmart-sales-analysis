from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model
import matplotlib.pyplot as plt
import time

# Load and preprocess data
print("Loading and preprocessing data...")
df = load_and_preprocess_data("/home/loay/walmart_sales_analysis/data/raw/Walmart Data Analysis and Forcasting.csv")
print("Data loaded and preprocessed.")

# Model training
print("Starting model training...")
start_time = time.time()
model, train_errors, test_errors, X_train, X_test, y_train, y_test = train_model(df)
end_time = time.time()
print(f"Model training completed in {end_time - start_time:.2f} seconds.")

# Plot error curves
print("Plotting training and testing error curves...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, 250), train_errors, label='Training Error')
plt.plot(range(1, 250), test_errors, label='Testing Error')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Error Curves')
plt.legend()
plt.savefig("/home/loay/walmart_sales_analysis/results/figures/training_testing_error_curves.png")
plt.show()
print("Error curves plotted and saved.")

# Evaluate model
print("Evaluating model performance...")
test_results = evaluate_model(model, X_train, y_train, X_test, y_test)
print("Model evaluation completed.")
print(f"Training R² Score: {test_results['train_r2']}")
print(f"Cross-Validated RMSE: {test_results['cv_rmse'].mean()} ± {test_results['cv_rmse'].std()}")

print("Training process finished. Model and results are ready.")
