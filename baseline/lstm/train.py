# train model

#imports
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from get_data_train_test import get_data
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from get_data_train_test import get_data_numpy

def train():
    
    # data 
    X_train, Y_train, X_test, Y_test = get_data_numpy()
    
    model=KerasClassifier(model=build_clf,layers=1,nodes=30,units=10,dropout=0.5)
    
    params={'batch_size':[32,64,128], 
        'epochs':[20,50,100],
        'layers':[1,2,3],
        'nodes':[30,60,90],
        'units':[5,10,50,100],
        'dropout':[0.2,0.3,0.5]
        }
    gs=GridSearchCV(estimator=model, param_grid=params,cv=10)
    # now fit the dataset to the GridSearchCV object. 
    gs = gs.fit(X_train, Y_train)
    
    print('best_params',gs.best_params_)
    print('accuracy',gs.best_score_)
    
def build_clf(layers,nodes,units,dropout):
  # creating the layers of the NN
    model = Sequential()
    
    if layers == 3:
        model.add(LSTM(nodes, input_shape=(30,8),activation='relu',return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(nodes,activation='relu',return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(nodes,activation='relu'))
        model.add(Dropout(dropout))
    elif layers == 2:
        model.add(LSTM(nodes, input_shape=(30,8),activation='relu',return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(nodes,activation='relu'))
        model.add(Dropout(dropout))
    else:
        model.add(LSTM(nodes, input_shape=(30,8)))
        model.add(Dropout(dropout))
        
    model.add(Dense(units, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    train()