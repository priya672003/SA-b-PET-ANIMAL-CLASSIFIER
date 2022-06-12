# SA-b-PET-ANIMAL-CLASSIFIER

# Algorithm
1. Import libraries required.
2. Load dataset through local or drive link.
3. Train the datasets.
4. Train the model with neural networks.
5. Compile the code.
6. Fit the model and check accuracy.
7. Evaluate performance on test dataset.

## Program:

Program to implement 

Developed by   : PRIYADARSHINI R

RegisterNumber :  212220230038

## OUTPUT:
1. CODE :

```PYTHON 3
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
X_train = np.loadtxt('input.csv', delimiter = ',')
Y_train = np.loadtxt('labels.csv', delimiter = ',')
X_test = np.loadtxt('input_test.csv', delimiter = ',')
Y_test = np.loadtxt('labels_test.csv', delimiter = ',')
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)
X_train = X_train/255.0
X_test = X_test/255.0
print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()
model = Sequential([
    Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32, (3,3), activation = 'relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 5, batch_size = 64)
model.evaluate(X_test, Y_test)
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()
y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5
if(y_pred == 0):
    pred = 'cat'
else:
    pred = 'dog'
    
print("Our model says it is a :", pred)

```

![image](https://user-images.githubusercontent.com/81132849/173227798-21dedf98-0c2e-4d6f-bca6-7a2368d5dce9.png)

![image](https://user-images.githubusercontent.com/81132849/173227819-ea19d22c-763f-4188-953b-91ce4d45687f.png)

![image](https://user-images.githubusercontent.com/81132849/173227846-3ccdfca0-609e-4831-9025-871c1e8d61f8.png)

![image](https://user-images.githubusercontent.com/81132849/173227872-1041c1fa-1c57-42f1-b419-eb0c333d16fc.png)



2. DEMO VIDEO YOUTUBE LINK:


