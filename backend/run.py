import os
import sys
from preprocessing import preprocess_data
from visualize import visualize_data
from train import train_models
from evaluate import evaluate_models

def main():
    try:
        # Bước 1: Tiền xử lý dữ liệu
        print("Bắt đầu tiền xử lý dữ liệu...")
        processed_data, base_dir, models_dir, plots_dir, logs_dir = preprocess_data()
        if processed_data is None:
            print("Lỗi trong quá trình tiền xử lý dữ liệu!")
            sys.exit(1)
        
        # Bước 2: Phân tích và vẽ biểu đồ
        print("\nBắt đầu phân tích và vẽ biểu đồ...")
        log_file = os.path.join(logs_dir, 'training_log.txt')
        if not visualize_data(processed_data, plots_dir, logs_dir, log_file):
            print("Lỗi trong quá trình phân tích và vẽ biểu đồ!")
            sys.exit(1)
        
        # Bước 3: Huấn luyện mô hình
        print("\nBắt đầu huấn luyện mô hình...")
        trained_models = train_models(processed_data, models_dir, logs_dir, log_file)
        if trained_models is None:
            print("Lỗi trong quá trình huấn luyện mô hình!")
            sys.exit(1)
        
        # Bước 4: Đánh giá và lưu mô hình
        print("\nBắt đầu đánh giá mô hình...")
        best_model_name, results = evaluate_models(trained_models, processed_data, 
                                                 models_dir, plots_dir, logs_dir, log_file)
        if best_model_name is None:
            print("Lỗi trong quá trình đánh giá mô hình!")
            sys.exit(1)
            
        print(f"\nQuá trình hoàn tất! Mô hình tốt nhất: {best_model_name}")
        print(f"Kết quả được lưu tại: {base_dir}")
        
    except Exception as e:
        print(f"Lỗi không mong muốn: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()