import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import missingno as msno


def show_missing_data(X, print_features):
    total_rows = len(X)
    missing_counts = X.isna().sum()
    missing_percents = missing_counts / total_rows

    missing_percents = missing_percents[missing_percents > 0].sort_values(ascending=False)
    top_missing = missing_percents.head(15)

    if top_missing.empty:
        print("No missing data found!")
        return

    if print_features:
        print("Missing values detected in:")
        print(missing_counts[missing_counts > 0])

    plt.figure(figsize=(8, 6))
    plt.barh(top_missing.index[::-1], top_missing.values[::-1],
             edgecolor='black', color='steelblue')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xlabel("Percent Missing")
    plt.title("Top 15 Columns with Most Missing Data")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

    msno.heatmap(X[top_missing.index], figsize=(8, 6), fontsize=10)
    plt.show()


def plot_corr(X, corr_threshold):
    corr_matrix = X.corr()

    # Remove weak correlations and self-correlations
    filtered_corr = corr_matrix.where((corr_matrix.abs() > corr_threshold) & (
        corr_matrix.abs() < 1.0)).dropna(how='all', axis=0).dropna(how='all', axis=1)

    plt.figure(figsize=(7.5, 6))
    sns.heatmap(filtered_corr, annot=True, fmt='.2f',
                cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()


def plot_label_corr(X, y):
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    top_corr = correlations[:10]

    plt.figure(figsize=(6, 4))
    plt.barh(top_corr.index, top_corr.values, edgecolor='black')
    plt.gca().invert_yaxis()
    plt.show()


def plot_pie(series):
    series = series.map({1: 'Default', 0: 'Repaid'})
    counts = series.value_counts()
    plt.figure(figsize=(4, 4))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
    plt.show()


def plot_histogram(X, column, bins=5, clip_quantile=None):
    X = X[column].dropna()

    if clip_quantile is not None:
        lower = X.quantile(clip_quantile[0])
        upper = X.quantile(clip_quantile[1])
        X = X[(X >= lower) & (X <= upper)]

    plt.figure(figsize=(4, 4))
    plt.hist(X, bins=bins, edgecolor='black')
    plt.show()
