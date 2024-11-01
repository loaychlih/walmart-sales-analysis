from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def train_model(df):
    X = df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Month', 'Year', 'Week']]
    y = df['Weekly_Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    gboost_reg = GradientBoostingRegressor(random_state=48)
    train_errors, test_errors = [], []

    for n_estimators in range(1, 250):
        gboost_reg.set_params(n_estimators=n_estimators)
        gboost_reg.fit(X_train, y_train)
        y_train_pred = gboost_reg.predict(X_train)
        y_test_pred = gboost_reg.predict(X_test)
        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))

    return gboost_reg, train_errors, test_errors, X_train, X_test, y_train, y_test
