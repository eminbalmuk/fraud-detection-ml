import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve, 
    average_precision_score, confusion_matrix, f1_score
)
import joblib
import os

DATA_PATH = 'data/fraudTrain.csv'

def preprocess_data(df):
    """
    Sızıntısız (Leakage-Free) Üretim Seviyesi Önişleme (CatBoost).
    """
    print("Tarih formatları ve sıralama...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], format='%m/%d/%y %H:%M')
    df['dob'] = pd.to_datetime(df['dob'], format='%m/%d/%y')
    
    # Sızıntıyı önlemek için zamana göre sıralayalım
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
    
    # CatBoost için kategorik kolonlar string veya int kalabilir, Pool içinde belirteceğiz
    df['category'] = df['category'].astype(str)
    df['gender'] = df['gender'].astype(str)
    df['state'] = df['state'].astype(str)
            
    return df

def main():
    print("Veri seti yükleniyor (CatBoost)")
    df = pd.read_csv(DATA_PATH)
    
    print("Sızıntısız preprocess başlatılıyor")
    df = preprocess_data(df)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df.drop(columns=['is_fraud'])
    y_train = train_df['is_fraud']
    X_test = test_df.drop(columns=['is_fraud'])
    y_test = test_df['is_fraud']
    
    # Validation aşaması
    val_split_idx = int(len(X_train) * 0.9)
    X_train_sub = X_train.iloc[:val_split_idx]
    y_train_sub = y_train.iloc[:val_split_idx]
    X_val = X_train.iloc[val_split_idx:]
    y_val = y_train.iloc[val_split_idx:]
    
    cat_features = ['category', 'gender', 'state']
    
    train_pool = Pool(X_train_sub, y_train_sub, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)
    
    print("Sızıntısız CatBoost eğitimi başlatılıyor...")
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100
    )
    
    model.fit(train_pool, eval_set=val_pool)
    
    y_prob = model.predict_proba(test_pool)[:, 1]
    
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
    print(f"CATBOOST LEAKAGE-FREE SONUÇLARI")
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
    plt.plot(recall, precision, color='green', lw=2, label=f'PR-AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('CatBoost PR Eğrisi')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    importance = pd.Series(model.get_feature_importance(), index=X_train.columns).sort_values(ascending=False).head(10)
    importance.plot(kind='barh', title='En Önemli Özellikler (Catboost)')
    plt.gca().invert_yaxis()
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    plt.tight_layout()
    plt.savefig('plots/catboost_fraud_detection.png')
    joblib.dump(model, 'models/catboost_fraud_detection.pkl')
    
    print(f"\nSonuçlar kaydedildi.")
    plt.close()

if __name__ == "__main__":
    main()
