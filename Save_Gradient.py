# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:24:19 2021
@author: Eduin Hernandez
"""
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import shelve
import numpy as np
from scipy import stats
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Cifar10 Training')

    'Model Details'
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size for Training and Testing')
    parser.add_argument('--epoch-num', type=int, default=100, help='End Iterations for Training')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for model')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    parser.add_argument('--model-name', type=str, default='NASNetMobile', help='Model to Load for Training')

    args = parser.parse_args()
    return args


# -----------------------------------------------------------------------------
def DenseNet121(input):
    return keras.applications.DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
    )


def DenseNet169(input):
    return keras.applications.DenseNet169(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
    )


def DenseNet201(input):
    return keras.applications.DenseNet201(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
    )


def ResNet50(input):
    return keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet101(input):
    return keras.applications.ResNet101(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet152(input):
    return keras.applications.ResNet152(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet50V2(input):
    return keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet101V2(input):
    return keras.applications.ResNet101V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet152V2(input):
    return keras.applications.ResNet152V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def NASNetMobile(input):
    return keras.applications.NASNetMobile(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10
    )


def NASNetLarge(input):
    return keras.applications.NASNetLarge(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10
    )


def VGG16(input):
    return keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax"
    )


def VGG19(input):
    return keras.applications.VGG19(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        #classifier_activation="softmax",
    )

tr_acc = tf.keras.metrics.CategoricalAccuracy()
tr_loss = tf.keras.metrics.CategoricalCrossentropy()

def step(X, y):
    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the
        # loss
        logits = model(X, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y,logits)
        tr_acc.update_state(y,logits)
        # Compute the loss and the initial gradient
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    """""""""
    for idx in range(len(grads)):
        temp = grads[idx].numpy()
        shape = np.shape(temp)
        grads[idx] = tf.convert_to_tensor(np.reshape(fp8_bin_centers[np.searchsorted(fp8_bin_edges,temp.flatten())],shape))
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return np.mean(loss.numpy())
    """""""""
    for idx in range(len(grads)):
        grads[idx] = grads[idx].numpy().flatten()
    return np.array(grads,dtype=object),np.mean(loss.numpy())


build_model = {'DenseNet121': DenseNet121,
               'DenseNet169': DenseNet169,
               'DenseNet201': DenseNet201,
               'ResNet50': ResNet50,
               'ResNet50V2': ResNet50V2,
               'ResNet101': ResNet101,
               'ResNet101V2': ResNet101V2,
               'ResNet152': ResNet152,
               'ResNet152V2': ResNet152V2,
               'NASNetMobile': NASNetMobile,
               'NASNetLarge': NASNetLarge,
               'VGG16': VGG16,
               'VGG19': VGG19
               }
# -----------------------------------------------------------------------------


def fp8_143_bin_edges(exponent_bias=10):
    bin_centers = np.zeros(239,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(239, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2, 2 ** -3],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(15):
            for f in range(8):
                if e != 0:
                    exponent = int(format(e, 'b').zfill(4), 2) - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(3)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(4) + format(f, 'b').zfill(3)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(3)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(4) + format(f,'b').zfill(3)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict["0.0"] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict

def fp8_152_bin_edges(exponent_bias=15):
    bin_centers = np.zeros(247,dtype=np.float32)
    fp8_binary_dict = {}
    fp8_binary_sequence = np.zeros(247, dtype='U8')
    binary_fraction = np.array([2 ** -1, 2 ** -2],dtype=np.float32)
    idx = 0
    for s in range(2):
        for e in range(31):
            for f in range(4):
                if e != 0:
                    exponent = int(format(e, 'b').zfill(5), 2) - exponent_bias
                    fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction)) + 1
                    bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                    fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(5) + format(f, 'b').zfill(2)
                    idx += 1
                else:
                    if f != 0:
                        exponent = 1-exponent_bias
                        fraction = np.sum((np.array(list(format(f, 'b').zfill(2)), dtype=int) * binary_fraction))
                        bin_centers[idx] = ((-1) ** (s)) * fraction * (2 ** exponent)
                        fp8_binary_dict[str(bin_centers[idx])] = str(s) + format(e, 'b').zfill(5) + format(f,'b').zfill(2)
                        idx += 1
                    else:
                        if s == 0:
                            bin_centers[idx] = 0
                            fp8_binary_dict["0.0"] = "00000000"
                            idx += 1
                        else:
                            pass
    bin_centers = np.sort(bin_centers)
    print(bin_centers)
    bin_edges = (bin_centers[1:] + bin_centers[:-1]) * 0.5
    return bin_centers, bin_edges, fp8_binary_dict

fp8_bin_centers,fp8_bin_edges,fp8_dict = fp8_152_bin_edges()
# -----------------------------------------------------------------------------
args = parse_args()
# -----------------------------------------------------------------------------
"Data Loading"
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# -----------------------------------------------------------------------------
"Data Preprocessing"
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# -----------------------------------------------------------------------------
"Model Creation"
model = build_model[args.model_name](x_train)

# initiate SGD optimizer
opt = keras.optimizers.SGD(learning_rate=args.learning_rate,
                           momentum=args.momentum)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()
# ------------------------------------------------------------------------------
"Data augmentation"
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
# -----------------------------------------------------------------------------
# compute the number of batch updates per epoch
batch_size = args.batch_size
epoch_num = args.epoch_num
numUpdates = int(x_train.shape[0] / batch_size)
train_acc = np.zeros(epoch_num)
test_acc = np.zeros(epoch_num)
train_loss = np.zeros(epoch_num)
test_loss = np.zeros(epoch_num)
filepath = "D:\\Zhong-Jing\\NASNetMobile\\"
epochStart = time.time()
for epoch in range(epoch_num):
    a0 = np.zeros(int(len(x_train) / batch_size) + 1)
    l0 = np.zeros(int(len(x_train) / batch_size) + 1)
    batches = 0
    for x_batch,y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
        grads,l1 = step(x_batch,y_batch)

        if batches == 0:
            avr_grads = np.zeros_like(grads)
        avr_grads += grads

        a1 = tr_acc.result().numpy()
        print("Training Accuracy: ",a1)
        a0[batches] = a1
        l0[batches] = l1
        batches += 1

        if batches >= len(x_train) / batch_size:
            my_shelf = shelve.open(filepath + 'Gradient_epoch' + str(epoch) + '.out')
            my_shelf['data'] = {'avr_grad': avr_grads/batches,'grads':grads}
            my_shelf.close()
            break

    t_loss, t_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("Epoch: " + str(epoch + 1) + '/' + str(epoch_num) + " Acc:" + str(f'{np.mean(a0): 0.3f}'))
    train_acc[epoch] = np.mean(a0)
    test_acc[epoch] = t_acc
    train_loss[epoch] = np.mean(l0)
    test_loss[epoch] = t_loss

epochEnd = time.time()
elapsed = (epochEnd - epochStart) / 60.0
print("took {:.4} minutes".format(elapsed))

my_shelf = shelve.open(filepath + 'Accuracy_and_loss' + '.out')
my_shelf['data'] = {'training_acc': train_acc, 'test_acc': test_acc,'training_loss':train_loss,'test_loss':test_loss}
my_shelf.close()
"""""""""
"Training"

hist = model.fit(x_train, y_train, batch_size=args.batch_size,
                 epochs=args.epoch_num, verbose=1,
                 validation_data=(x_test, y_test),
                 shuffle=True)

accuracy = hist.history['accuracy']
loss = hist.history['loss']

accuracy_val = hist.history['val_accuracy']
loss_val = hist.history['val_loss']
# -----------------------------------------------------------------------------
"Plotting"
plt.close('all')
plt.figure(1)
plt.title('Accuracy')
plt.plot(accuracy)
plt.plot(accuracy_val)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'])

plt.figure(2)
plt.title('Loss')
plt.plot(loss)
plt.plot(loss_val)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'])
"""""""""
