# Plotting

get_ipython().run_line_magic('matplotlib', 'inline')

def plot_histogram(data, col, title, xlabel, xmin, xmax, save_as, style='whitegrid', font_scale=1.2):
    '''
    This function plots the histogram of a specific column/ feature in the data
    '''
    sns.set_style(style=style)
    sns.set(font_scale=font_scale
    sns.distplot(a = data[col], kde = False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.axis(xmin=xmin, xmax=xmax)
    plt.savefig(save_as)