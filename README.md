# Image-similarity-for-face-recognition

match.py is the python script that takes paths of two input images as input and give the Match or No Match with the confidence of similarity.
for example 
python3 match.py -p1 path1 -p2 path2 
where path1 and path2 are your image paths 

The model.py contains the architecture of the model used for similarity checking and the model file contains the model with the parameters.

The image_similarity_model.ipynb is the notebook containing all the steps to train and test the model with proper comments.
I worked on notebook in google colabs and at some places I have saved and loaded the data form google drive. The file on google drive is open to all so no need to change that. Only the path may need to change if you are using any other platform than colabs.
