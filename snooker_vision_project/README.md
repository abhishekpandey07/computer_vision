# Snooker_Vision_Project
This project uses machine learning and image processing to detect and classify balls on a snooker table using an angled camera.
The corresponding youtube video used for processing can be found here: https://www.youtube.com/watch?v=lRvYTfVVSlc.
On linux command:

youtube-dl https://www.youtube.com/watch?v=lRvYTfVVSlc
can be used to download the video into the current directory.

The coding format makes extensive use of pipelines modules from sklearn. The file work_frame_pipeline.py reads the video file and performs computation and classification.

The dataset/balls/balls/ directory consists of images of snooker balls in 'labeled' folders. pipeline_draft.py uses pipelines to extract these images and performs the passed computations and feature extraction on the dataset and saves the dataset into the folder dataset/balls/preprocessedDataPipeline folder using cPickle module

Some pretrained classifiers can  be found in predictors folders.

The dataset can be processed in ovo ( colour classification ) and ova ( ball/ not ball for false positives) mode. The classifiers have been trained and saved on each of these modes and with various preprocessing indicated by the prefix of file  names.

hue  --> HSV colorspace ( only h vector)

gray --> grayscale images

hog --> HOG descriptors
