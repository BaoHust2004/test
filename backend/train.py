import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
import joblib
import json
import datetime

def log_message(message, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def train_models(processed_data, models_dir, logs_dir, log_file):
    try:
        log_message("===== BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH =====", log_file)
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        log_message(f"Kích thước tập huấn luyện: {X_train.shape[0]} mẫu, {X_train.shape[1]} đặc trưng", log_file)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42)
        }
        
        log_message(f"Số lượng mô hình sẽ huấn luyện: {len(models)}", log_file)
        
        param_grids = {
            'Linear Regression': {
                'fit_intercept': [True, False],
                'copy_X': [True, False],
                'positive': [True, False]
            },
            'Decision Tree': {
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        }
        
        # Train và optimize mỗi mô hình
        trained_models = {}
        for name, model in models.items():
            log_message(f"\nBắt đầu huấn luyện mô hình {name}...", log_file)
            log_message(f"Tham số tìm kiếm: {param_grids[name]}", log_file)
            
            grid = GridSearchCV(
                model,
                param_grids[name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            log_message(f"Đang thực hiện Grid Search với 5-fold cross validation...", log_file)
            grid.fit(X_train, y_train)
            
            log_message(f"Kết quả tốt nhất cho {name}:", log_file)
            log_message(f"- Best parameters: {grid.best_params_}", log_file)
            log_message(f"- Best score (neg MSE): {grid.best_score_:.4f}", log_file)
            
            trained_models[name] = grid.best_estimator_
            
            # Lưu mô hình
            model_path = os.path.join(models_dir, f'{name.lower().replace(" ", "_")}.pkl')
            joblib.dump(grid.best_estimator_, model_path)
            log_message(f"Đã lưu mô hình tại: {model_path}", log_file)
        
        log_message("\n===== HOÀN THÀNH HUẤN LUYỆN MÔ HÌNH =====", log_file)
        return trained_models
        
    except Exception as e:
        log_message(f"Lỗi trong quá trình huấn luyện mô hình: {str(e)}", log_file)
        return None

if __name__ == "__main__":
    pass