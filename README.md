# BEAN-SNN: Spiking Deep CNN trained with Biologically-Enhanced Artificial Neuronal Assembly Regularization

The main.py is the original bean regularization code from the repository https://github.com/YuyangGao/BEAN.
Here we added a small script at the end to save the model in serial format in Keras.
The conversion is performed using onyx.
The Pytorch model is first converted to intermediate onyx format.
Thereafter, the onyx format is converted into Keras.
The evaluate.py script evaluates the Keras model to verify the accuracy is same as in the original Pytorch model.
The accuracy of LeNet with BEAN regularization in Pytorch and Keras is 99.15%.

All package dependencies are indicated in the requirements.txt file.
