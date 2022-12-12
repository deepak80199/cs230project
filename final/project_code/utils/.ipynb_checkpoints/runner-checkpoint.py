# main 
from utils import preprocessing
from utils import classifier
from utils import evaluate

def analyse():
    path_to_video = 'path'
    
    df_user=preprocessing.preprocess(path_to_video)
    pose = classifier.classify(df_user)
    
    evaluate.evaluate_pose(df_user,pose,path_to_video)