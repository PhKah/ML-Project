import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Load data
df = pd.read_csv('Data/Speed Dating Data.csv', encoding='ISO-8859-1')

# 2. Định nghĩa danh sách biến
# Bao gồm các biến nhân khẩu học, sở thích (preferences) và chấm điểm (ratings)
essential_vars = ['gender', 'age', 'age_o', 'race', 'imprace', 'imprelig', 
                  'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 
                  'attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'condtn', 'match']

df_subset = df[essential_vars].copy()

# Phân loại cột để xử lý điền khuyết (Imputation)
numerical_cols = ['age', 'age_o', 'imprace', 'imprelig', 
                  'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 
                  'attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
categorical_cols = ['race', 'gender', 'condtn']

# 3. Xử lý giá trị khuyết thiếu (Missing Values)
# Sử dụng Median cho biến số để tránh ảnh hưởng của giá trị ngoại lai
df_subset[numerical_cols] = df_subset[numerical_cols].fillna(df_subset[numerical_cols].median())

# Sử dụng Mode cho biến phân loại
for col in categorical_cols:
    df_subset[col] = df_subset[col].fillna(df_subset[col].mode()[0])

# 4. Feature Engineering (Age Gap)
# Tính chênh lệch tuổi tuyệt đối giữa hai đối tượng
df_subset['age_diff'] = abs(df_subset['age'] - df_subset['age_o'])
# Loại bỏ age_o vì đã được trích xuất vào age_diff
df_subset.drop('age_o', axis=1, inplace=True)

# 5. Mã hóa biến phân loại (Encoding)
# Sử dụng One-hot encoding cho sắc tộc (race)
df_encoded = pd.get_dummies(df_subset, columns=['race'], prefix='race', drop_first=True)

# Lưu tập dữ liệu đã làm sạch nhưng chưa chuẩn hóa để đối soát
df_encoded.to_csv('Data/data_cleaned.csv', index=False)
print("--- Step 1-3: Cleaned and encoded data saved to Data/data_cleaned.csv ---")

# 6. Chuẩn hóa dữ liệu (Standardization)
# Đưa các biến số về cùng thang đo (Z-score: Mean=0, Std=1)
scaler = StandardScaler()
# Cập nhật danh sách cột cần scale (loại bỏ age_o và thêm age_diff)
cols_to_scale = [c for c in numerical_cols if c != 'age_o'] + ['age_diff']

df_final = df_encoded.copy()
df_final[cols_to_scale] = scaler.fit_transform(df_final[cols_to_scale])

# 7. Lưu tập dữ liệu cuối cùng sẵn sàng cho Modeling
df_final.to_csv('Data/data_final.csv', index=False)
print(f"--- Step 4: Final standardized data saved to Data/data_final.csv. Shape: {df_final.shape} ---")
