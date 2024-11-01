from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = (-cv_scores) ** 0.5

    y_test_pred_final = model.predict(X_test)
    test_results = {
        "train_r2": train_r2,
        "cv_rmse": cv_rmse,
        "test_predictions": y_test_pred_final
    }
    return test_results
