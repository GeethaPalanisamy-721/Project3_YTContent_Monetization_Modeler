import matplotlib.pyplot as plt
import seaborn as sns
import os

def basic_summary(df):
    print(df.describe(include='all'))
    print('\nMissing values:')
    print(df.isna().sum())

def plot_target_distribution(df, target='ad_revenue_usd', save_dir='artifacts'):
    os.makedirs(save_dir, exist_ok=True)
    df[target].hist(bins=50)
    plt.title('Target distribution')
    save_path = os.path.join(save_dir, f"{target}_distribution.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def correlation_heatmap(df, save_dir='artifacts'):
    os.makedirs(save_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_path = os.path.join(save_dir, "correlation_heatmap.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()
