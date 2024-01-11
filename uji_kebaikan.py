import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Membaca dataset
df = pd.read_csv('smoking_driking_dataset_Ver01.csv')

# Analisis dengan 10 variabel numerik menggunakan Python
X_full = df[['age', 'height', 'weight', 'waistline', 'hemoglobin', 'triglyceride', 'serum_creatinine', 'BLDS', 'HDL_chole', 'LDL_chole']]
y_full = df['SBP']

# Menambahkan konstanta untuk termasuk intercept
X_full = sm.add_constant(X_full)

# Membuat model regresi
model_full = sm.OLS(y_full, X_full).fit()

# Menampilkan hasil regresi
print("Hasil Regresi untuk 10 Variabel Numerik:")
print(model_full.summary())

# Menghitung R-squared
r_squared = model_full.rsquared

# Menampilkan hasil uji R-squared
print(f"\nUji Kebaikan Model (R-squared):")
print(f"R-squared: {r_squared:.4f}")

# Interpretasi
if r_squared == 1:
    print("Model dapat menjelaskan seluruh variasi dalam data.")
elif r_squared > 0.7:
    print("Model memiliki kemampuan yang baik untuk menjelaskan variasi dalam data.")
elif r_squared > 0.5:
    print("Model memiliki kemampuan moderat untuk menjelaskan variasi dalam data.")
else:
    print("Model memiliki kemampuan rendah untuk menjelaskan variasi dalam data.")
