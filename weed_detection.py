import os
import cv2
import numpy as np
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical
import matplotlib.pyplot as plt


class WeedCNN:
    def __init__(self, batch_size = 32, epochs = 10, lr= 0.001, img_dims = (64,64,3), image_dims2D = (64,64), regularizerValue = 0.0001):
        self.images = []
        self.labels = []
        self.val_images = []
        self.val_labels = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = None
        self.num_classes = 2
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.img_dims = img_dims
        self.image_dims2D = image_dims2D
        self.regularizerValue = regularizerValue

    def classify_image(self, image_path):
        label = image_path.split(os.path.sep)[-2]
        if label == "noweed":
            label = 0
        else:
            label = 1
        return label
    
    def load_images(self, path):
        img_list = []
        for subdir, dirs, files in os.walk(path):
            for file in files:
                putanja_fajla = subdir + os.sep + file
                if putanja_fajla.endswith(".jpg") or putanja_fajla.endswith(".jpeg"):
                    img_list.append(putanja_fajla)
        return img_list


    def load_train_data(self):
        train_lista = self.load_images("./dataset/train")
        np.random.shuffle(train_lista)
        for img in train_lista:
            image = cv2.imread(img)
            image = cv2.resize(image, self.image_dims2D)
            self.images.append(image)
            label = self.classify_image(img)
            self.labels.append([label])

        self.train_images = np.array(self.images) / 255.0
        self.train_labels = np.array(self.labels)
        self.train_labels = to_categorical(self.train_labels, self.num_classes)
        print("Number of train images:", len(self.train_images))
        print("Number of train labels:", len(self.train_labels))

    def load_validation_data(self):
        val_lista = self.load_images("./dataset/validation")
        np.random.shuffle(val_lista)
        for img in val_lista:
            image = cv2.imread(img)
            image = cv2.resize(image, self.image_dims2D)
            self.val_images.append(image)
            label = self.classify_image(img)
            self.val_labels.append([label])

        self.val_images = np.array(self.val_images) / 255.0
        self.val_labels = np.array(self.val_labels)
        self.val_labels = to_categorical(self.val_labels, self.num_classes)
        print("Number of validation images:", len(self.val_images))
        print("Number of validation labels:", len(self.val_labels))
    
    def load_test_data(self):
        test_lista = self.load_images("./dataset/test")
        np.random.shuffle(test_lista)
        for img in test_lista:
            image = cv2.imread(img)
            image = cv2.resize(image, self.image_dims2D)
            self.test_images.append(image)
            label = self.classify_image(img)
            self.test_labels.append([label])

        self.test_images = np.array(self.test_images) / 255.0
        self.test_labels = np.array(self.test_labels)
        self.test_labels = to_categorical(self.test_labels, self.num_classes)
        print("Number of test images:", len(self.test_images))
        print("Number of test labels:", len(self.test_labels))

    def build_model1(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.img_dims, kernel_regularizer=l2(self.regularizerValue)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self.img_dims, kernel_regularizer=l2(self.regularizerValue)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))


    def train_model(self):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(self.train_images)

        opt = Adam(learning_rate=0.0001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        h = self.model.fit(
            datagen.flow(self.train_images, self.train_labels, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(self.val_images, self.val_labels),
        )

        #self.plot_training(h)

    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_images,
                                                       self.test_labels)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    def predict_weed(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_dims2D)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:
            return 'no weed'
        elif predicted_class == 1:
            return 'weed'

        return "prediction error"

    def plot_training(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'weedCNN-lr_{self.lr}-bs_{self.batch_size}-ep_{self.epochs}.png')
        plt.draw()