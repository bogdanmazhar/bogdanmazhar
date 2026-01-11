from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = 'binary' if num_classes == 2 else 'categorical'
train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, rotation_range=20, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(128, 128),
    batch_size=32,
    class_mode=class_mode,
    subset='validation'
)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),
    BatchNormalization(),
    MaxPooling2D(),
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') if class_mode == 'binary' else Dense(num_classes, activation='softmax')
])

loss_function = 'binary_crossentropy' if class_mode == 'binary' else 'categorical_crossentropy'
model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
callbacks = [
EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3)
]
model.fit(train_data, validation_data=val_data, epochs=30, callbacks=callbacks)
test_loss, test_accuracy = model.evaluate(val_data)
print(f'Точность модели на валидационных данных: {test_accuracy:.2f}')