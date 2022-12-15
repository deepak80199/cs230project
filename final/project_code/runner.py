# main 
from utils import preprocessing
from utils import classifier
from utils import evaluate
import sys


def analyse(file_path):
    """
    Process user video

    Parameters:
    file_path (string): Path to the user pose vide

    Returns:
    None
    """
    path_to_video = file_path

    # preprocess the video to get user angle timeseries dataframe
    df_user = preprocessing.preprocess(path_to_video)

    # classify the user pose
    pose = classifier.classify(df_user)

    # compare the user pose with instructor pose
    evaluate.evaluate_pose(df_user, pose, path_to_video)


if __name__ == "__main__":
    analyse(sys.argv[0])
