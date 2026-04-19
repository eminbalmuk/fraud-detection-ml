import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_eda():
    print("Veri seti yükleniyor...")
    # Sadece gerek duyulan sütunları çekerek bellek tasarrufu yapalım
    cols = ['trans_date_trans_time', 'category', 'amt', 'is_fraud']
    df = pd.read_csv('data/fraudTrain.csv', usecols=cols)

    # Tarih dönüşümü
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour

    print("Grafikler oluşturuluyor...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle('Kredi Kartı Dolandırıcılık Tespiti - Veri Seti Analizi', fontsize=20, fontweight='bold')

    # Sınıf Dağılımı (Fraud vs Normal)
    fraud_counts = df['is_fraud'].value_counts()
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Dolandırıcılık Sınıf Dağılımı (0: Normal, 1: Fraud)', fontsize=14)
    axes[0, 0].set_yscale('log') # İnanılmaz dengesizlik olduğu için logaritmik ölçek
    for i, v in enumerate(fraud_counts.values):
        axes[0, 0].text(i, v, f'{v}\n({v/len(df)*100:.2f}%)', ha='center', va='bottom', fontweight='bold')

    # Harcama Miktarları Dağılımı (Log-Scale)
    sns.histplot(data=df, x='amt', hue='is_fraud', kde=True, ax=axes[0, 1], palette='magma', bins=50)
    axes[0, 1].set_title('İşlem Tutarları Dağılımı (Log-Scale)', fontsize=14)
    axes[0, 1].set_xlim(0, 2000) # Okunabilirlik için limit
    axes[0, 1].set_yscale('log')

    # Kategori Bazlı Dolandırıcılık Oranları
    cat_fraud = df.groupby('category')['is_fraud'].mean().sort_values(ascending=False)
    sns.barplot(x=cat_fraud.values, y=cat_fraud.index, ax=axes[1, 0], palette='rocket')
    axes[1, 0].set_title('Kategorilere Göre Dolandırıcılık Oranı', fontsize=14)
    axes[1, 0].set_xlabel('Fraud Oranı')

    # Saatlik İşlem Dağılımı
    hourly_counts = df.groupby(['hour', 'is_fraud']).size().unstack()
    hourly_counts_norm = hourly_counts.div(hourly_counts.sum(axis=0), axis=1) # Normalize ederek trendi görelim
    hourly_counts_norm.plot(kind='line', ax=axes[1, 1], marker='o')
    axes[1, 1].set_title('Saatlik İşlem Yoğunluğu (Normalize Edilmiş Trend)', fontsize=14)
    axes[1, 1].set_xlabel('Günün Saati')
    axes[1, 1].set_ylabel('Yoğunluk')
    axes[1, 1].legend(['Normal', 'Fraud'])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Kayıt yeri
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/dataset_analysis.png', dpi=300)
    print("Analiz grafiği 'plots/dataset_analysis.png' olarak kaydedildi.")
    plt.close()

if __name__ == "__main__":
    generate_eda()
