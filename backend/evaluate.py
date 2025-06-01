import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import json
import joblib
import datetime

def log_message(message, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def evaluate_models(trained_models, processed_data, models_dir, plots_dir, logs_dir, log_file):
    try:
        log_message("===== BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH =====", log_file)
        
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        log_message(f"Kích thước tập kiểm tra: {X_test.shape[0]} mẫu, {X_test.shape[1]} đặc trưng", log_file)
        log_message(f"Số lượng mô hình cần đánh giá: {len(trained_models)}", log_file)
        
        results = {}
        best_rmse = float('inf')
        best_model = None
        best_model_name = None
        
        for name, model in trained_models.items():
            log_message(f"\nĐang đánh giá mô hình {name}...", log_file)
            y_pred = model.predict(X_test)
            
            # Tính các metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            log_message(f"Kết quả đánh giá {name}:", log_file)
            log_message(f"- RMSE: {rmse:.4f}", log_file)
            log_message(f"- MAE: {mae:.4f}", log_file)
            log_message(f"- R2 Score: {r2:.4f}", log_file)
            
            # Vẽ biểu đồ dự đoán vs thực tế
            log_message(f"Đang vẽ biểu đồ dự đoán vs thực tế cho {name}...", log_file)
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.title(f'{name}: Predicted vs Actual', fontsize=14)
            plt.xlabel('Actual', fontsize=12)
            plt.ylabel('Predicted', fontsize=12)
            plot_path = os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_predictions.png')
            plt.savefig(plot_path)
            plt.close()
            log_message(f"Đã lưu biểu đồ tại: {plot_path}", log_file)
            
            # Cập nhật mô hình tốt nhất
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = name
                log_message(f"Đã cập nhật mô hình tốt nhất: {name} (RMSE: {rmse:.4f})", log_file)
        
        # Lưu kết quả đánh giá
        eval_path = os.path.join(logs_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=4)
        log_message(f"Đã lưu kết quả đánh giá tại: {eval_path}", log_file)
        
        # Lưu mô hình tốt nhất
        best_model_path = 'models/best_model.pkl'
        joblib.dump(best_model, best_model_path)
        log_message(f"Đã lưu mô hình tốt nhất ({best_model_name}) tại: {best_model_path}", log_file)
        
        log_message("\n===== HOÀN THÀNH ĐÁNH GIÁ MÔ HÌNH =====", log_file)
        log_message(f"Mô hình tốt nhất: {best_model_name}", log_file)
        log_message(f"- RMSE: {results[best_model_name]['RMSE']:.4f}", log_file)
        log_message(f"- MAE: {results[best_model_name]['MAE']:.4f}", log_file)
        log_message(f"- R2 Score: {results[best_model_name]['R2']:.4f}", log_file)
        
        return best_model_name, results
        
    except Exception as e:
        log_message(f"Lỗi trong quá trình đánh giá mô hình: {str(e)}", log_file)
        return None, None

if __name__ == "__main__":
    pass