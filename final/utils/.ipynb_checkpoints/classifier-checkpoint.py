# classifier
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np 

def classify(user_df):
    num  = user_df.shape[0]//30
    new_df = pd.dataframe


    # selecting 30 equidistant points
    k=0
    
    for j in range(30):
        new_df = pd.concat([new_df,data.iloc[k].to_frame().T],ignore_index=True)
        k=k+num
        
    new_df = tf.convert_to_tensor(new_df)
    
    new_df = new_df/360
    
    
    model = keras.models.load_model('model/v2-97.91')
    yhat = model.predict(new_df)
    yhat = np.argmax(yhat)
    return yhat