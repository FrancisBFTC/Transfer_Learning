# Importação das bibliotecas
import os
import zipfile
import random

import tensorflow as tf
import subprocess
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# Download do dataset
# !wget --no-check-certificate \
#  "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip" \
#  -O "/content/sample_data/cats-and-dogs.zip"

# Efetua o Download e Extração do Dataset
url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
file_name = "/content/sample_data/cats-and-dogs.zip"

subprocess.run(["powershell", "-Command", f"Invoke-WebRequest -Uri {url} -OutFile {file_name}"])
print(f"Arquivo salvo como {file_name}")

local_zip = '/content/sample_data/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/sample_data')
zip_ref.close()

# Imprime a quantidade de arquivos do dataset
main_dir = '/content/sample_data/'
print(len(os.listdir(main_dir + 'PetImages/Cat/')))
print(len(os.listdir(main_dir + 'PetImages/Dog/')))

# Cria as pastas de treinamento e teste
try:
  os.mkdir(main_dir + 'cats-v-dogs')
  os.mkdir(main_dir + 'cats-v-dogs/training')
  os.mkdir(main_dir + 'cats-v-dogs/testing')
  os.mkdir(main_dir + 'cats-v-dogs/training/cats')
  os.mkdir(main_dir + 'cats-v-dogs/training/dogs')
  os.mkdir(main_dir + 'cats-v-dogs/testing/cats')
  os.mkdir(main_dir + 'cats-v-dogs/testing/dogs')
except OSError:
  pass

# Função para mover dataset para pastas de treino e teste
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

# Executa a cópia do dataset para as pastas de treino e teste
CAT_SOURCE_DIR = main_dir + "PetImages/Cat/"
TRAINING_CATS_DIR = main_dir + "cats-v-dogs/training/cats/"
TESTING_CATS_DIR = main_dir + "cats-v-dogs/testing/cats/"

DOG_SOURCE_DIR = main_dir + "PetImages/Dog/"
TRAINING_DOGS_DIR = main_dir + "cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = main_dir + "cats-v-dogs/testing/dogs/"

split_size = 9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Imprima o tamanho de dados das pastas de treino e teste
print(len(os.listdir(TRAINING_CATS_DIR)))
print(len(os.listdir(TRAINING_DOGS_DIR)))
print(len(os.listdir(TESTING_CATS_DIR)))
print(len(os.listdir(TESTING_DOGS_DIR)))

# Cria o modelo de classificação de imagens
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = main_dir + "cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=250,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
VALIDATION_DIR = main_dir + "cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=250,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')
plt.show

