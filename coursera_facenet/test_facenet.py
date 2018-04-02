import cv2
import glob
from fr_utils import load_database
from facenet_inspired_facerecognition import *

FRmodel = initialise_model()
database = load_database(FRmodel)
print "DATABASE: {}".format(database.keys())

for i in glob.glob('../test_images/*'):
    image = cv2.imread(i, 1)
    who_is_it(image, database, FRmodel)
