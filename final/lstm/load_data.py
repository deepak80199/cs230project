# load data

# imports
import pandas as pd
import os
    
def load_base_data():
    #{'Bhuj': 1 ,'Padam':2,  'Shav':3 , 'Tada':4 , 'Trik':5  ,'Vriksh':6}
    # print(dir_list)
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