import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_directories():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'training_results_{timestamp}'
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')
    logs_dir = os.path.join(base_dir, 'logs')
    
    for directory in [models_dir, plots_dir, logs_dir, 'models']:
        os.makedirs(directory, exist_ok=True)
    
    return base_dir, models_dir, plots_dir, logs_dir

def log_message(message, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def preprocess_data(data_path='../csv/train.csv'):
    try:
        base_dir, models_dir, plots_dir, logs_dir = setup_directories()
        log_file = os.path.join(logs_dir, 'training_log.txt')
        
        log_message("===== BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU =====", log_file)
        
        # Đọc dữ liệu
        data = pd.read_csv(data_path)
        log_message(f"Đọc dữ liệu thành công: {data.shape[0]} dòng và {data.shape[1]} cột", log_file)
        
        # Xử lý giá trị null
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            log_message("Bắt đầu xử lý giá trị null...", log_file)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(exclude=[np.number]).columns
            
            for col in numeric_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].mean(), inplace=True)
                    log_message(f"Đã điền giá trị null trong cột {col} bằng giá trị trung bình", log_file)
            
            for col in categorical_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].mode()[0], inplace=True)
                    log_message(f"Đã điền giá trị null trong cột {col} bằng giá trị phổ biến nhất", log_file)
        
        # Mã hóa biến phân loại
        log_message("Bắt đầu mã hóa biến phân loại...", log_file)
        data_encoded = pd.get_dummies(data, drop_first=True)
        log_message(f"Số đặc trưng sau khi mã hóa: {data_encoded.shape[1]}", log_file)
        
        # Chia dữ liệu
        X = data_encoded.drop('G3', axis=1)
        y = data_encoded['G3']
        
        # Chia tập train/test
        log_message("Chia tập train/test với tỉ lệ 80/20...", log_file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Chuẩn hóa dữ liệu
        log_message("Bắt đầu chuẩn hóa dữ liệu...", log_file)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Lưu scaler
        joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
        joblib.dump(scaler, 'models/scaler.pkl')
        log_message("Đã lưu scaler", log_file)
        
        processed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'feature_names': X.columns.tolist(),
            'original_data': data
        }
        
        log_message("===== HOÀN THÀNH TIỀN XỬ LÝ DỮ LIỆU =====", log_file)
        return processed_data, base_dir, models_dir, plots_dir, logs_dir
        
    except Exception as e:
        log_message(f"Lỗi trong quá trình tiền xử lý: {str(e)}", log_file)
        raise

if __name__ == "__main__":
    preprocess_data()