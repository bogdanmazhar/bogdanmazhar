import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
model = tf.keras.models.load_model('fashion_mnist_model.h5')
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255
x_test = x_test.reshape(-1, 28, 28)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_image(index):
    img = x_test[index]
    img_expanded = img.reshape(1, 28, 28)
    predictions = model.predict(img_expanded)
    predicted_class = predictions[0].argmax()
    confidence = predictions[0][predicted_class] * 100
    plt.imshow(img, cmap='gray')
    plt.title(f'Предсказанно: {class_names[predicted_class]} ({confidence:.2f}%)\n Истинный класс: {class_names[y_test[index]]}\n')
    plt.axis('off')
    plt.show()

random_index = random.randint(0, len(x_test) - 1)
predict_image(random_index)