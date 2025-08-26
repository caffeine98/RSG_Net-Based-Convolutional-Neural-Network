# RSG-Net implementation to detect Diabetic Retinopathy

This project is an update to my Pytorch capstone project from Codecademy (https://github.com/caffeine98/Codecademy-Pytorch-Capstone-Image-Classification), where I built 2 different models. With the hope of getting an improved result, I decided to implement a model based on the RSG-Net (Retinopathy Severity Grading Net) Convolutional Neural Network model, that was proposed in the paper "A deep learning based model for diabetic retinopathy grading" by Samia Akhtar, Shabib Aftab, Oualid Ali, Munir Ahmad, Muhammad Adnan Khan, Sagheer Abbas & Taher M.Ghazal. The paper can be found here: https://www.nature.com/articles/s41598-025-87171-9

# RSG-Net Architecture

This model consists of four convolutional layers, two max pooling layers, one flatten layer, one fully connected layer, 5 batch normalization layers and one dropout layer. The original paper adds a batch normalization after the flattened layer, but I decided to add it to the convolutional layers as well. Max pooling is applied after the second and the fourth convlutional layers. All the layers uses the ReLU activation function, except for the last layer, which uses the sigmoid activation function.

# Dataset

The dataset used to train the model is from Indian Diabetic Retinopathy Image Dataset (IDRiD), which consists of 516 fundus images. This dataset is split into 413 training images and 103 testing images.

# Training

The model uses a Stochastic Gradient Descent optimizer, with a learning rate of 0.001. Binary Cross Entropy loss function is used since it is a binary classification task. This model was trained for 5 epochs.

# Results

After 5 epochs, the training and validation losses were in the range of 0.58 to 0.59. After testing the model with the test dataset, a classification report was generated.

```
                                precision   recall     f1-score   support

Does not have retinopathy           0.42      0.29      0.34        34
          Has retinopathy           0.70      0.80      0.74        69

                 accuracy                               0.63       103
                macro avg           0.56      0.55      0.54       103
             weighted avg           0.60      0.63      0.61       103
```
The model had an accuracy score of 63%. More importantly, it was able to detect images that did not have retinopathy 34% of the time. This is a significantly better result than those from the two models in my Codecademy capstone project, as those models failed to detect images that did not have retinopathy (The first model might not have been complex enough, while the second VGG-16 based model might have been too complex and led to overfitting). Overall, I would say that this is a better model than the ones in my capstone project.

Possible improvements include using a larger dataset. The paper from which I implemented the RSG-Net model, used the Messidor-1 dataset, which is a much larger dataset of 1200 colour fundus images, compared to the IDRiD dataset, that had 516 images. Also running for higher epochs could help train and improve the model further.
