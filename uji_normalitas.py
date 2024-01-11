import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson

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

# Uji Normalitas (Shapiro-Wilk)
residuals = model_full.resid
shapiro_stat, shapiro_p_value = shapiro(residuals)
print("\nShapiro-Wilk Test for Normality of Residuals:")
print(f"Test Statistic: {shapiro_stat}, P-Value: {shapiro_p_value:.10f}")
if shapiro_p_value < 0.05:
    print("Residuals tidak terdistribusi normal.")
else:
    print("Residuals terdistribusi normal.")

# Uji Heteroskedastisitas (White Test)
white_stat, white_p_value, _, _ = het_white(residuals, exog=X_full)
print("\nWhite Test for Heteroscedasticity:")
print(f"Test Statistic: {white_stat}, P-Value: {white_p_value:.10f}")
if white_p_value < 0.05:
    print("Model mengalami heteroskedastisitas.")
else:
    print("Model tidak mengalami heteroskedastisitas.")

# Uji Autokorelasi (Durbin-Watson)
dw_stat = durbin_watson(residuals)
print("\nDurbin-Watson Test for Autocorrelation of Residuals:")
print(f"Test Statistic: {dw_stat}")
if dw_stat < 1.5 or dw_stat > 2.5:
    print("Residuals tidak menunjukkan adanya autokorelasi.")
else:
    print("Residuals menunjukkan adanya autokorelasi.")
