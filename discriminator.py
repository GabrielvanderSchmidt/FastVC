import os

import numpy as np
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D, \
    LeakyReLU
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import RMSprop
from keras import backend as K

import cv2


# noinspection PyCallingNonCallable
def feat_extractor(input_size: tuple = (224, 224, 1), conv_layers: tuple = (32, 16, 8), ksize: int = 3,
                   maxpool_size: int = 3, dense_layers: tuple = (256,), embedding_size: int = 512,
                   dropout_rate: float = 0.1, conv_activation: object = "relu",
                   dense_activation: object = "sigmoid") -> Model:
    inputs = Input(shape=input_size)
    x = inputs
    for filters in conv_layers:
        x = Conv2D(filters=filters, kernel_size=ksize, padding="valid")(x)
        if isinstance(conv_activation, str):  # If conv_activation is not a string, then it must be a function
            x = Activation(conv_activation)(x)
        else:
            x = conv_activation(x)
        x = BatchNormalization()(x)
        if not (maxpool_size is None or maxpool_size == 0):
            x = MaxPool2D(pool_size=maxpool_size)(x)
        x = Dropout(rate=dropout_rate)(x)
    x = Flatten()(x)
    for neurons in dense_layers:
        if isinstance(dense_activation, str):  # If dense_activation is not a string, then it must be a function
            x = Dense(units=neurons, activation=dense_activation)(x)
        else:
            x = Dense(units=neurons)(x)
            x = dense_activation(x)

    if isinstance(dense_activation, str):
        outputs = Dense(units=embedding_size, activation=dense_activation)(x)
    else:
        x = Dense(units=embedding_size)(x)
        outputs = dense_activation(x)

    extractor = Model(inputs=inputs, outputs=outputs)
    return extractor


def load_images(path: str) -> tuple:
    """
    Load all images from path directory and apply some data augmentation. Warning: ALL images (loaded and augmented) are
    kept in memory.
    :param path: Path to dataset dir
    :return: X, y tuple
    """
    X = []
    y = []
    print("Loading images...")
    for label in os.listdir(path):
        if not os.path.isdir(os.path.join(path, label)):
            continue
        for file in os.listdir(os.path.join(path, label)):
            if not (os.path.isfile(os.path.join(path, label, file)) and file.endswith(".png")):
                continue
            image = cv2.imread(os.path.join(path, label, file), cv2.IMREAD_GRAYSCALE)
            assert not (image is None)
            flipped = cv2.flip(image, 1)  # Flip horizontally
            images = [cv2.add(image, cv2.randn(image.copy(), (0,), (30,))), cv2.add(flipped, cv2.randn(flipped.copy(), (0,), (30,)))]
            X.extend(images) # Original image + noise & Flipped image + noise
            y.extend([label, label])
    print("Images loaded!")
    return np.array(X), np.array(y)


###  Code below from https://stackoverflow.com/a/41645732 (with some minor changes)
class DataGenerator(object):
    """docstring for DataGenerator"""

    def __init__(self, batch_sz):
        root = r"/home/schmidt/GitHub/FastVC/LibriSpeech/spectrograms"
        X_train, y_train = load_images(root)
        X_train = X_train.astype('float32')
        X_train /= 255

        # create training+test positive and negative pairs
        self.keys = np.unique(y_train)
        speaker_indices = dict((key, np.where(y_train == key)[0]) for key in self.keys)
        self.tr_pairs, self.tr_y = self.create_pairs(X_train, speaker_indices)

        #speaker_indices = [np.where(y_test == i)[0] for i in range(10)]
        #self.te_pairs, self.te_y = self.create_pairs(X_test, speaker_indices)

        self.tr_pairs_0 = self.tr_pairs[:, 0]
        self.tr_pairs_1 = self.tr_pairs[:, 1]
        #self.te_pairs_0 = self.te_pairs[:, 0]
        #self.te_pairs_1 = self.te_pairs[:, 1]

        self.batch_sz = batch_sz
        self.samples_per_train = (self.tr_pairs.shape[0] / self.batch_sz) * self.batch_sz
        #self.samples_per_val = (self.te_pairs.shape[0] / self.batch_sz) * self.batch_sz

        self.cur_train_index = 0
        #self.cur_val_index = 0

    def create_pairs(self, x, speaker_indices):
        '''Positive and negative pair creation.
        Alternates between positive and negative pairs.
        '''
        pairs = []
        labels = []
        n = min([len(speaker_indices[key]) for key in self.keys]) - 1
        for d in range(10):
            for i in range(n):
                z1, z2 = speaker_indices[d][i], speaker_indices[d][i + 1]
                pairs += [[x[z1], x[z2]]]
                inc = random.randrange(1, 10)
                dn = (d + inc) % 10
                z1, z2 = speaker_indices[d][i], speaker_indices[dn][i]
                pairs += [[x[z1], x[z2]]]
                labels += [1, 0]
        return np.array(pairs), np.array(labels)

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            yield ([self.tr_pairs_0[self.cur_train_index:self.cur_train_index + self.batch_sz],
                    self.tr_pairs_1[self.cur_train_index:self.cur_train_index + self.batch_sz]
                    ],
                   self.tr_y[self.cur_train_index:self.cur_train_index + self.batch_sz]
                   )

    # def next_val(self):
    #     while 1:
    #         self.cur_val_index += self.batch_sz
    #         if self.cur_val_index >= self.samples_per_val:
    #             self.cur_val_index = 0
    #         yield ([self.te_pairs_0[self.cur_val_index:self.cur_val_index + self.batch_sz],
    #                 self.te_pairs_1[self.cur_val_index:self.cur_val_index + self.batch_sz]
    #                 ],
    #                self.te_y[self.cur_val_index:self.cur_val_index + self.batch_sz]
    #                )


# In the future, I should replace this code with a custom DataGenerator, but right now I can't be bothered, I just want
# this thing to run.
# Maybe something like https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

###  Also from https://stackoverflow.com/a/41645732
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# His example was a classification task. Since that is not our case, we won't need an accuracy metric.
# def compute_accuracy(predictions, labels):
#     '''Compute classification accuracy with a fixed threshold on distances.
#     '''
#     return labels[predictions.ravel() < 0.5].mean()

print("Setting up model...")
input_size = (224, 224, 1)
nb_epoch = 10
batch_size = 128

datagen = DataGenerator(batch_size)

# Combine parts to make the network
extractor = feat_extractor(input_size=(224, 224, 1), conv_layers=(8, 8, 8, 8), ksize=5, maxpool_size=3, dense_layers=[], conv_activation=LeakyReLU(alpha=0.3))
image_a = Input(shape=input_size)
image_b = Input(shape=input_size)

# Weights are shared between the two halves by using the same instance of feat_extractor.
x_a = extractor(image_a)
x_b = extractor(image_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([x_a, x_b])

model = Model(input=[image_a, image_b], output=distance)
print(model.summary())
# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
print("Initiating training!")
model.fit_generator(generator=datagen.next_train(), samples_per_epoch=datagen.samples_per_train, nb_epoch=nb_epoch)

#model = feat_extractor()
print(model.summary())
print("Done!")
