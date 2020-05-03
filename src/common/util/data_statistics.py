# compute preliminary statistics information of a dataset

def data_statistics(data):
    '''
    function that gets the statistics information on data
    '''
    number_nan = [data[col].isnull().sum() for col in data.columns]
    number_distinct = [data[col].nunique() for col in data.columns]
    data_type = [type(data[col][0]) for col in data.columns]
    data_summary = pd.DataFrame({'columns': data.columns,
                                 'number_nan': number_nan,
                                 'number_distinct': number_distinct,
                                'data_type': data_type})
    data_summary.set_index('columns')
    return data_summary