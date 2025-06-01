import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import datetime  # Thêm import datetime

def visualize_data(processed_data, plots_dir, logs_dir, log_file):
    try:
        log_message("===== BẮT ĐẦU PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU =====", log_file)
        data = processed_data['original_data']
        
        # Phân tích biến mục tiêu G3
        log_message("Đang vẽ biểu đồ phân phối điểm số G3...", log_file)
        plt.figure(figsize=(10, 6))
        sns.histplot(data['G3'], kde=True, bins=20)
        plt.title('Phân phối điểm số G3', fontsize=14)
        plt.xlabel('Điểm số', fontsize=12)
        plt.ylabel('Tần suất', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'G3_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        log_message("Đã lưu biểu đồ phân phối G3", log_file)
        
        # Phân tích tương quan
        log_message("Đang tính toán ma trận tương quan...", log_file)
        numeric_data = data.select_dtypes(include=['number'])
        correlation = numeric_data.corr()
        correlation.to_csv(os.path.join(logs_dir, 'correlation_matrix.csv'))
        log_message(f"Đã lưu ma trận tương quan với {len(numeric_data.columns)} biến số", log_file)
        
        # Vẽ heatmap tương quan
        log_message("Đang vẽ heatmap tương quan...", log_file)
        plt.figure(figsize=(14, 12))
        mask = np.triu(correlation)
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", 
                    linewidths=0.5, mask=mask)
        plt.title('Ma trận tương quan', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        log_message("Đã lưu heatmap tương quan", log_file)
        
        # Phân tích các biến phân loại
        categorical_columns = data.select_dtypes(include=['object']).columns
        log_message(f"\nBắt đầu phân tích {len(categorical_columns)} biến phân loại...", log_file)
        
        for col in categorical_columns:
            log_message(f"Đang phân tích biến {col}...", log_file)
            plt.figure(figsize=(10, 6))
            avg_by_category = data.groupby(col)['G3'].mean().sort_values(ascending=False)
            
            # Ghi thống kê về biến
            category_stats = data[col].value_counts()
            log_message(f"Thống kê biến {col}:", log_file)
            log_message(f"- Số lượng giá trị unique: {len(category_stats)}", log_file)
            log_message(f"- Giá trị phổ biến nhất: {category_stats.index[0]} ({category_stats.values[0]} lần)", log_file)
            
            sns.barplot(x=avg_by_category.index, y=avg_by_category.values, palette='viridis')
            plt.title(f'Điểm trung bình G3 theo {col}', fontsize=14)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'G3_by_{col}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            log_message(f"Đã lưu biểu đồ phân tích biến {col}", log_file)
        
        log_message("===== HOÀN THÀNH PHÂN TÍCH VÀ TRỰC QUAN HÓA DỮ LIỆU =====", log_file)
        return True
        
    except Exception as e:
        log_message(f"Lỗi trong quá trình phân tích và vẽ biểu đồ: {str(e)}", log_file)
        return False

def log_message(message, log_file):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

if __name__ == "__main__":
    pass