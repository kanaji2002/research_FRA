# 必要なライブラリのインポート
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# データの読み込み
data = pd.read_csv('power.csv')

# トレーニングデータとテストデータに分割
train_data, test_data = train_test_split(data, test_size=0.1, random_state=0)

# 特徴量とターゲットの分割
X_train = train_data.drop(columns=['MUsage1', 'MUsage2', 'Tem', 'GPU_Uti', 'M_Uti', 'CommonP', 'model1_P_2pir', 'model2_P_2pair', 'model_py', 'P_Usage', 'model1_FLOPS'])
y_train = train_data['model1+2_P']
X_test = test_data.drop(columns=['MUsage1', 'MUsage2', 'Tem', 'GPU_Uti', 'M_Uti', 'CommonP', 'model1_P_2pir', 'model2_P_2pair', 'model_py', 'P_Usage', 'model1_FLOPS'])
y_test = test_data['model1+2_P']

# 1層目のモデルの構築と予測
lgbm = LGBMRegressor(random_state=0)
rf = RandomForestRegressor(random_state=0)
gbr = GradientBoostingRegressor(random_state=0)

# 1層目モデルの学習
lgbm.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# 1層目モデルの予測結果を取得
lgbm_pred = lgbm.predict(X_train)
rf_pred = rf.predict(X_train)
gbr_pred = gbr.predict(X_train)

# 1層目の予測結果を特徴量として2層目に入力
stacked_features_train = np.column_stack((lgbm_pred, rf_pred, gbr_pred))
stacked_features_test = np.column_stack((
    lgbm.predict(X_test),
    rf.predict(X_test),
    gbr.predict(X_test)
))

# 2層目のモデル（線形回帰）の学習と予測
lr = LinearRegression()
lr.fit(stacked_features_train, y_train)
stacked_pred = lr.predict(stacked_features_test)

# MSEの評価
mse = mean_squared_error(y_test, stacked_pred)
print(f"Stacking Model MSE: {mse}")


from sklearn.metrics import confusion_matrix

# 実測値と予測値を1.0と2.0で3段階に分類
def categorize_chl_a(values):
    return np.where(values < 1.0, 0, np.where(values < 2.0, 1, 2))

# 実測値と予測値をカテゴリに分ける
y_test_category = categorize_chl_a(y_test)
stacked_pred_category = categorize_chl_a(stacked_pred)

# 混合行列の計算と表示
cm = confusion_matrix(y_test_category, stacked_pred_category)
print("Confusion Matrix:\n", cm)
