# Mineral-Prospectivity-Prediction-Convolutional-Neural-Networks
Considering that our data of mineral prospectivity prediction is private, we randomly selected 10 samples as an example to show the mineral prospectivity modeling process and guide you to operate the codes. The used 10 samples are in the file folder of "sample" in "Data".

Before run, please copy the file folder of "sample" and renamed it as "train".

Data augmentation.py:
Divide the dataset into the training and validation sets that 80% of the samples for training and 20% of the samples for validation, and utilize the data augmentation method to generate more samples. The file folder of "train_expand" contains the samples for training, and the file folder of "verify_expand" contains the samples for validation. Then, transform samples into the format of “datalist”.

data_loader_channels.py:
Read the file of “datalist” to get multi-channel samples and their labels for inputting into convolutional neural network (CNN).

LeNet.py: LeNet structure of CNN.

LeNet_main.py: Train the LeNet model and verify the performance of the LeNet model. 

AlexNet.py: AlexNet structure of CNN.

AlexNet_main.py: Train the AlexNet model and verify the performance of the AlexNet model. 

VggNet.py: VggNet structure of CNN.

VggNet_main.py: Train the VggNet model and verify the performance of the VggNet model.
