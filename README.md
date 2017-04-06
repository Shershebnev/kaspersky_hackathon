Code for Kaspersky hackathon https://events.kaspersky.com/hackathon/  

The aim of the online part of the hackathon was to detect timepoints of attack on a plant using multivariable time series data from various sensors. The attack is supposed to produce anomalies in sensors' measurements.  
The proposed solution consists of two parts: recurrent neural network, built using two LSTM layers and one fully connected output regression layer (with linear activation function), and used for prediction of values with a given initial small set of observations. Predicted and actual values were then compared to each other using MSE over a sliding window with the idea that the window with highest MSE will contain the attack point.

Tested using:  
Python 3.6.0, Numpy 1.12.1, Pandas 0.19.2, Scikit-learn 0.18.1, TensorFlow 1.0.1, Keras 2.0.2  

Training and test dataset available here:  
https://events.kaspersky.com/hackathon/uploads/kaspersky_hackathon_1.zip  
(also in this repository)

The work up to the hackathon deadline was done under assumption of independency of different columns. However, most likely there is some interplay between measurements of different samples hence it would be interesting to try to predict all features at once rather than using distinct RNN for each feature separately. Doing this also involved an intersting (for a DL newbie, at least) of resolving the issue of limited memory size - the whole training dataset wouldn't fit into memory when split into batches. Initial solution involved splitting training dataset into several files, however this involved (depending on the split parameters) heavy disk I/O operations, which slowed down the whole analysis pipeline significantly. Moreover it involved extra steps for random shuffling of samples. Another approach was used instead: it is sufficient to store pairs of indexes of batch start and end rows, which can than later be used both for shuffling and extracting subsets of training dataset. With this it was pretty trivial to write a generator for use in `model.fit_generator()` function, as well as `model.predict_generator`. The Neural Network was also modified to reflect the changes in the number of input and output neurons. These tweaks allowed to use all features simultaneously for training and prediction. The respective code will be pushed in couple days.
