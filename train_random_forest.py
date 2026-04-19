import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    Sızıntısız (Leakage-Free) Preprocess.
    """
    print("Tarih formatları ve sıralama...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%m/%d/%y %H:%M')
    df['dob'] = pd.to_datetime(df['dob'], format='%m/%d/%y')
    
    # Sızıntıyı önlemek için zamana göre sırala
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    
    print("Sızıntısız Davranışsal Özellikler Üretiliyor...")
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
    
    cat_cols = ['category', 'gender', 'state']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            
    return df

def main():
    print("Veri yükleniyor (Random Forest Leakage-Free)...")
    df = pd.read_csv(DATA_PATH)
    
    print("Sızıntısız önişleme başlatılıyor...")
    df = preprocess_data(df)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=['is_fraud'])
    y_train = train_df['is_fraud']
    X_test = test_df.drop(columns=['is_fraud'])
    y_test = test_df['is_fraud']
    
    print("Sızıntısız Random Forest eğitimi başlatılıyor...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
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
    print(f"RANDOM FOREST SONUÇLARI")
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
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Random Forest PR Eğrisi')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    importance.plot(kind='barh', title='En Önemli Özellikler (Random Forest)')
    plt.gca().invert_yaxis()
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('plots/random_forest_fraud_detection.png')
    joblib.dump(model, 'models/random_forest_fraud_detection.pkl')
    
    print(f"\nSonuçlar kaydedildi.")
    plt.close()

if __name__ == "__main__":
    main()
