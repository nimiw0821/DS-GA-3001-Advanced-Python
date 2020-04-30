import pandas as pd

def load_file(file_path, col_select_list = None):
    '''
    load a file from directory
    '''
    file = pd.read_csv(file_path, header=0, usecols = col_select_list)
    return file