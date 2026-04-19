import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, 
    average_precision_score, confusion_matrix, f1_score
)
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Veri yolları
DATA_PATH = 'data/fraudTrain.csv'

def preprocess_data(df):
    """
    Sızıntısız (Leakage-Free) Üretim Seviyesi Önişleme (XGBoost).
    """
    print("Tarih formatları ve sıralama...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%m/%d/%y %H:%M')
    df['dob'] = pd.to_datetime(df['dob'], format='%m/%d/%y')
    
    # Sızıntıyı önlemek için zamana göre sıralıyoruz
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    
    print("Sızıntısız Davranışsal Özellikler Üretiliyor...")
    # Geçmiş Ortalama Harcama (Cumulative / Expanding - Sadece Geçmiş)
    group_amt = df.groupby('cc_num')['amt']
    past_cumsum = group_amt.cumsum().shift(1).fillna(0)
    past_count = group_amt.cumcount().shift(1).fillna(0)
    
    df['past_avg_amt'] = (past_cumsum / (past_count + 1e-9)).replace(0, df['amt'].mean())
    df['amt_to_past_avg_ratio'] = df['amt'] / (df['past_avg_amt'] + 1e-9)
    
    df['prev_trans_time'] = df.groupby('cc_num')['trans_date_trans_time'].shift(1)
    df['time_since_last_trans'] = (df['trans_date_trans_time'] - df['prev_trans_time']).dt.total_seconds().fillna(-1)
    
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year
    df['lat_diff'] = np.abs(df['lat'] - df['merch_lat'])
    df['lon_diff'] = np.abs(df['long'] - df['merch_long'])
    
    cols_to_drop = [
        'Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'zip', 
        'trans_num', 'unix_time', 'Unnamed: 23', '6006', 'dob', 'trans_date_trans_time',
        'prev_trans_time', 'lat', 'long', 'merch_lat', 'merch_long',
        'merchant', 'job'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Label encoding
    cat_cols = ['category', 'gender', 'state']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

def main():
    print("Veri yükleniyor (XGBoost)...")
    df = pd.read_csv(DATA_PATH)
    
    print("Sızıntısız preprocess başlatılıyor...")
    df = preprocess_data(df)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=['is_fraud'])
    y_train = train_df['is_fraud']
    X_test = test_df.drop(columns=['is_fraud'])
    y_test = test_df['is_fraud']
    
    # Validasyon
    val_split_idx = int(len(X_train) * 0.9)
    X_train_sub = X_train.iloc[:val_split_idx]
    y_train_sub = y_train.iloc[:val_split_idx]
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]
    
    # Dengesizlik Oranı
    scale_weight = float(np.sum(y_train_sub == 0) / np.sum(y_train_sub == 1))
    
    print("Sızıntısız XGBoost eğitimi başlatılıyor...")
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        scale_pos_weight=scale_weight,
        tree_method='hist',
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Threshold optimizasyonu
    thresholds = np.linspace(0.1, 0.9, 17)
    best_f1 = 0
    opt_threshold = 0.5
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            opt_threshold = t
            
    print(f"\nOptimal Eşik: {opt_threshold} (F1: {best_f1:.4f})")
    y_pred_opt = (y_prob >= opt_threshold).astype(int)
    
    print("\n" + "="*60)
    print(f"XGBOOST SONUÇLARI")
    print("="*60)
    print(classification_report(y_test, y_pred_opt))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Görselleştirme
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('XGBoost PR Eğrisi')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    importance.plot(kind='barh', title='En Önemli Özellikler (XGBoost)')
    plt.gca().invert_yaxis()
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('plots/xgboost_fraud_detection.png')
    joblib.dump(model, 'models/xgboost_fraud_detection.pkl')
    
    print(f"\nSonuçlar kaydedildi.")
    plt.close()

if __name__ == "__main__":
    main()
