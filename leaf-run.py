#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python

import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

import argparse

parser = argparse.ArgumentParser(description='Perform leaf training.')
parser.add_argument('model', metavar='model',
                   help='CNN or R50 for ResNet50.')
parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                   help='number of epchs for training.')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                   help='run a test on the model.')
parser.add_argument('--id', dest='testimage',
                   help='run a test on a single image.')

args = parser.parse_args()

epochs = args.epochs
runmodel = args.model
runtest = args.test
testimage = args.testimage

import datetime
now = datetime.datetime.now().isoformat()
print("START",now)

logdir = "logs/%s-%s" % (runmodel, now)

K.set_image_dim_ordering('tf')
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

image_size=224

batch_size=64
lrate = 0.01

train_dir = "data/train"
validation_dir = "data/validation"
test_dir = "data/test"

train_datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

validation_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

test_datagen = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical')

num_classes = len(train_generator.class_indices)
input_shape = train_generator.image_shape

print("CLASSES")

classes_dict = {}
for name, val in train_generator.class_indices.items():
  classes_dict[val] = name
classes = []
for i in range(0, num_classes):
  print(i, classes_dict[i])
  classes.append(classes_dict[i])

K.clear_session()

if runtest:
    model = keras.models.load_model("leaf.%s.h5" % runmodel)
    evaltest =  model.evaluate_generator(test_generator)
    for name, val in zip(model.metrics_names, evaltest):
        print(name, val)
    exit(0)

if testimage is not None:
    img = cv2.imread(testimage, cv2.IMREAD_GRAYSCALE)
    img = np.reshape(img, (1, 224, 224, 1))
    model = keras.models.load_model("leaf.%s.h5" % runmodel)
    ptest =  model.predict(img)
    for i in np.argsort(-ptest)[0][:3]:
        print(i, classes[i], ptest[0, i])
    exit(0)

if runmodel == "R50":
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(input_shape=input_shape, weights=None, classes=num_classes)
elif runmodel == "MNET":
    from keras.applications.mobilenet import MobileNet
    model = MobileNet(input_shape=input_shape, weights=None, classes=num_classes)
elif runmodel == "CNN2":
    model = keras.models.load_model("leaf.CNN.h5")
elif runmodel == "CNN":
# Create the model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='softmax'))
else:
    print("Invalid model")
    exit(1)

model.summary()

print("TRAINING PHASE")

decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True)
tensorboard.set_model(model)

history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[tensorboard])

model.save("leaf.%s.h5" % runmodel)

print("TESTING PHASE")

evaltest =  model.evaluate_generator(test_generator, 1)
for name, val in zip(model.metrics_names, evaltest):
    print(name, val)

print("END", datetime.datetime.now().isoformat())
