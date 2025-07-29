import matplotlib.pyplot as plt
import seaborn as sns



def plot_pie(df, column, title=None):
    if title == None: title = column
    df[column].value_counts(dropna=False).plot(
        kind='pie',
        autopct='%1.1f%%',
        ylabel='',
        title=title,
        figsize=(4, 4)
    )
    plt.show()



def plot_histogram(df, column, title=None, xlabel=None, ylabel='Frequency', bins=10):
    if title == None: title = column
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



def plot_bar(df, column, title=None, xlabel=None, ylabel='Count'):
    if title == None: title = column
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



def plot_box(df, column, by, title=None, xlabel=None, ylabel=None):
    if title == None: title = column
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