import pandas as pd

def load_file(file_path):
    '''
    load a file from directory
    '''
    file = pd.read_csv(file_path, header=0)
    return file