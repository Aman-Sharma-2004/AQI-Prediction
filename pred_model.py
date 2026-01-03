import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
import warnings 
import os 
warnings.filterwarnings('ignore') 
os.chdir(r"C:\Users\promo\OneDrive\Desktop\Data_Science\tf_env") 
air_data = pd.read_csv("aqidaily_combined.csv") 
air_data['Date'] = pd.to_datetime(air_data['Date']) 
air_data['month'] = air_data['Date'].dt.month 
print(air_data.head()) 
air_data['day'] = air_data['Date'].dt.day 
le_pollutant = LabelEncoder() 
air_data['day_of_week'] = air_data['Date'].dt.dayofweek 
plt.style.use('seaborn-v0_8-darkgrid') 
print(air_data.describe()) 
air_data['pollutant_encoded'] 
Pollutant']) 
f
ig = plt.figure(figsize=(18, 10)) 
plt.subplot(2, 3, 1) 
= 
le_pollutant.fit_transform(air_data['Main 
plt.hist(air_data['Overall AQI Value'], bins=40, color='steelblue', edgecolor='black', 
alpha=0.7) 
plt.title('AQI Value Distribution') 
plt.xlabel('AQI Value') 
plt.ylabel('Frequency') 
le_site = LabelEncoder() 
air_data['site_encoded'] = le_site.fit_transform(air_data['Site Name (of Overall 
AQI)']) 
plt.subplot(2, 3, 3) 
pollutant_count = air_data['Main Pollutant'].value_counts() 
plt.bar(pollutant_count.index, 
edgecolor='black', alpha=0.8) 
pollutant_count.values, 
plt.title('Primary Pollutant Distribution') 
plt.xlabel('Pollutant Type') 
plt.ylabel('Count') 
plt.xticks(rotation=45) 
le_source = LabelEncoder() 
plt.subplot(2, 3, 2) 
plt.plot(air_data['Date'], 
color='darkblue') 
plt.title('AQI Trend') 
plt.xlabel('Date') 
plt.ylabel('AQI Value') 
plt.xticks(rotation=45) 
air_data['Overall 
AQI 
Value'], 
color='coral', 
linewidth=0.9, 
air_data['source_encoded'] = le_source.fit_transform(air_data['Source (of Overall 
AQI)']) 
plt.subplot(2, 3, 5) 
plt.scatter(air_data['PM25'], air_data['Overall AQI Value'], alpha=0.5, s=30, 
color='green') 
plt.title('PM2.5 vs AQI') 
plt.xlabel('PM2.5') 
plt.ylabel('AQI') 
num_cols = ['Overall AQI Value', 'CO', 'Ozone', 'PM10', 'PM25', 'NO2', 'month', 'day', 
'day_of_week'] 
plt.subplot(2, 3, 6) 
air_data.boxplot(column='Overall AQI Value', by='month') 
plt.title('AQI by Month') 
plt.suptitle('') 
plt.xlabel('Month') 
plt.ylabel('AQI') 
corr = air_data[num_cols].corr() 
plt.subplot(2, 3, 4) 
sns.heatmap(corr, 
annot=True, 
square=True, linewidths=0.5) 
plt.title('Correlation Matrix') 
plt.tight_layout() 
plt.show() 
fmt='.2f', 
cmap='coolwarm', 
center=0, 
feat = ['CO', 'Ozone', 'PM10', 'PM25', 'NO2', 'month', 'day', 'day_of_week', 
'pollutant_encoded', 'site_encoded', 'source_encoded'] 
x = air_data[feat] 
y = air_data['Overall AQI Value'] 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42) 
rf 
= 
RandomForestRegressor(n_estimators=100, 
max_depth=20, 
min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1) 
rf.fit(xtrain, ytrain) 
pred_train = rf.predict(xtrain) 
pred_test = rf.predict(xtest) 
rmse1 = np.sqrt(mean_squared_error(ytrain, pred_train)) 
r2_1 = r2_score(ytrain, pred_train) 
mae1 = mean_absolute_error(ytrain, pred_train) 
rmse2 = np.sqrt(mean_squared_error(ytest, pred_test)) 
r2_2 = r2_score(ytest, pred_test) 
print(rmse1, mae1, r2_1) 
mae2 = mean_absolute_error(ytest, pred_test) 
print(rmse2, mae2, r2_2) 
imp_df = pd.DataFrame({'Feature': feat, 'Importance': rf.feature_importances_}) 
imp_df = imp_df.sort_values('Importance', ascending=False) 
print(imp_df) 
f, axes = plt.subplots(2, 3, figsize=(16, 10)) 
axes[0, 0].barh(imp_df['Feature'], imp_df['Importance'], color='teal', alpha=0.8) 
axes[0, 0].set_xlabel('Importance Score') 
axes[0, 0].set_title('Feature Importance') 
axes[0, 0].invert_yaxis() 
axes[0, 2].scatter(pred_test, ytest - pred_test, alpha=0.6, s=30, color='red') 
axes[0, 2].axhline(y=0, color='black', linestyle='--', lw=2) 
axes[0, 2].set_xlabel('Predicted AQI') 
axes[0, 2].set_ylabel('Residuals') 
axes[0, 2].set_title('Residual Analysis') 
axes[0, 1].scatter(ytest, pred_test, alpha=0.6, s=30, color='blue') 
axes[0, 1].plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'r--', lw=2) 
axes[0, 1].set_xlabel('Actual AQI') 
axes[0, 1].set_ylabel('Predicted AQI') 
axes[0, 1].set_title(f'Test Set Predictions (R² = {r2_2:.3f})') 
residual = ytest - pred_test 
axes[1, 1].scatter(ytrain, pred_train, alpha=0.4, s=20, color='green') 
axes[1, 1].plot([ytrain.min(), ytrain.max()], [ytrain.min(), ytrain.max()], 'r--', lw=2) 
axes[1, 1].set_xlabel('Actual AQI') 
axes[1, 1].set_ylabel('Predicted AQI') 
axes[1, 1].set_title(f'Train Set Predictions (R² = {r2_1:.3f})') 
axes[1, 0].hist(residual, bins=40, color='lightcoral', edgecolor='black', alpha=0.7) 
axes[1, 0].set_xlabel('Prediction Error') 
axes[1, 0].set_ylabel('Frequency') 
axes[1, 0].set_title('Error Distribution') 
metric_label = ['RMSE', 'MAE', 'R²'] 
vals_train = [rmse1, mae1, r2_1 * 100] 
vals_test = [rmse2, mae2, r2_2 * 100] 
positions = np.arange(len(metric_label)) 
width = 0.35 
axes[1, 2].bar(positions - width/2, vals_train, width, label='Train', color='lightblue', 
alpha=0.8) 
axes[1, 2].bar(positions + width/2, vals_test, width, label='Test', color='salmon', 
alpha=0.8) 
axes[1, 2].set_ylabel('Values') 
axes[1, 2].set_title('Performance Metrics') 
axes[1, 2].set_xticks(positions) 
axes[1, 2].set_xticklabels(metric_label) 
axes[1, 2].legend() 
plt.tight_layout() 
plt.show() 
sample = pd.DataFrame({'Actual': ytest.values[:15], 'Predicted': pred_test[:15], 
'Error': ytest.values[:15] - pred_test[:15]}) 
sample['Absolute_Error'] = sample['Error'].abs() 
print(sample)