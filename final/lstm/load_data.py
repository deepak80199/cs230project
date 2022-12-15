# load data

# imports
import pandas as pd
import os
    
def load_base_data():
    """
    This class is used to load data from csv files

    Parameters:
        None

    Returns:
        x : Dataframe of angle time series
        y : Label(yoga pose) corresponding to each record in x
    """
    x = []
    y = []
    dir = ['Bhuj','Padam','Shav','Tada','Trik','Vriksh']
    for i in dir:
        dir_list = os.listdir('../Yoga_data_large_labeled/'+i)
        if '.ipynb_checkpoints' in dir_list:
            dir_list.remove('.ipynb_checkpoints')
        for j in dir_list:
            data =  pd.read_csv('../Yoga_data_large_labeled/'+i+'/'+j)
            y.append(data['yoga'][0])
            data = data.drop('yoga',axis=1)
            x.append(data.values)
    return x,y