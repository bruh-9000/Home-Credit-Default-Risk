import matplotlib.pyplot as plt
import seaborn as sns



def plot_pie(df, column, title='Pie Chart'):
    df[column].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        ylabel='',
        title=title,
        figsize=(4, 4)
    )
    plt.show()



def plot_histogram(df, column, title='Histogram', xlabel=None, ylabel='Frequency', bins=10):
    df[column].plot(
        kind='hist',
        bins=bins,
        edgecolor='black',
        title=title,
        xlabel=xlabel or column,
        ylabel=ylabel,
        figsize=(4, 4)
    )
    plt.show()



def plot_bar(df, column, title='Bar Chart', xlabel=None, ylabel='Count'):
    df[column].value_counts().sort_index().plot(
        kind='bar',
        edgecolor='black',
        title=title,
        xlabel=xlabel or column,
        ylabel=ylabel,
        figsize=(4, 4)
    )
    plt.show()



def plot_line(df, x, y, title='Line Plot', xlabel=None, ylabel=None):
    df.plot(
        kind='line',
        x=x,
        y=y,
        title=title,
        xlabel=xlabel or x,
        ylabel=ylabel or y,
        figsize=(4, 4)
    )
    plt.show()



def plot_box(df, column, by, title='Box Plot', xlabel=None, ylabel=None):
    df.boxplot(
        column=column,
        by=by,
        figsize=(4, 4)
    )
    plt.title(title)
    plt.suptitle('')
    plt.xlabel(xlabel or by)
    plt.ylabel(ylabel or column)
    plt.show()



def plot_scatter(df, x, y, title='Scatter Plot', xlabel=None, ylabel=None):
    df.plot(
        kind='scatter',
        x=x,
        y=y,
        title=title,
        figsize=(4, 4)
    )
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.show()



def plot_correlation_heatmap(corr_matrix, title='Feature Correlation Heatmap'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(title)
    plt.show()