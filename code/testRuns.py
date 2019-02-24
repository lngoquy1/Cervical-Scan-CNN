from time import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from glob import glob

import os
from numpy import sqrt, histogram
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import np_utils
#from sklearn.multiclass import OneVsRestClassifer

from preprocessing import *
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
np.random.seed(123)  # for reproducibility

from numpy.lib.stride_tricks import as_strided
import scipy.cluster.vq as vq
from sklearn.utils import shuffle

### MAIN
def main():
    X, Y, min_dim = getData()
    X, Y = shuffle(X, Y, random_state=0)
    #X_submit, dim_submit = getSubmitData()
    X_traintune, X_test, Y_traintune, Y_test = train_test_split(X, Y, test_size=0.33)
    
    CNNplot(X_traintune, Y_traintune, 32, 8, 0, min_dim) #1
    X_train, X_tune, Y_train, Y_tune = train_test_split(X_traintune, Y_traintune, test_size=.2)
    BOWS(X_train, Y_train, X_test, Y_test, 5, 400)
    MLPR(X_train, Y_train, X_test, Y_test)
    #CNNplot(X_traintune, Y_traintune, 16, 8, 0, min_dim)
    #CNNplot(X_traintune, Y_traintune, 16, 8, 2, min_dim)
    #CNNplot(X_traintune, Y_traintune, 16, 16, 0, min_dim) #2
    #CNNplot(X_traintune, Y_traintune, 32, 8, 0, min_dim) #3
    #CNNplot(X_traintune, Y_traintune, 32, 8, 1, min_dim) #4
    #CNNplot(X_traintune, Y_traintune, 32, 8, 2, min_dim) #5
    #CNNplot(X_traintune, Y_traintune, 64, 8, 3, min_dim) #8
    #CNNplot(X_traintune, Y_traintune, 32, 8, 1, min_dim) #9 redo with F = 5
    #CNNplot(X_traintune, Y_traintune, 32, 8, 2, min_dim) #10 redo with F = 11
    #CNNplot(X_traintune, Y_traintune, 32, 8, 4, min_dim) #9 redo
    #CNN(X_train, Y_train, X_test, Y_test, min_dim)
    #MLPR(X_train, Y_train, X_test, Y_test)
    #print "X_train", X_train.shape
    #print "X_test", X_train.shape
    #print "X_submit", X_submit.shape
    print "===========================================\n"

   
def MLPR(X_train, Y_train, X_test, Y_test):
    """
    Our multi-layer perceptron method using a single layer with 16 nodes
    """

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(16), max_iter=50, alpha=1e-4,\
        solver='adam', activation='relu', verbose=0, tol=1e-4, random_state=1)
    start_time = time()
    mlp.fit(X_train, Y_train)
    print "time:", time()-start_time
    print("\n Train set score %f\n" % mlp.score(X_train, Y_train))
    print("\nTest set score: %f\n" % mlp.score(X_test, Y_test))



def BOWS(X_train, Y_train, X_test, Y_test, P, K):
    """
    Our implementation for Bags of Visual Words
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    # Getting the patches
    patches, size = view_patches(X_train, P)
    print "Patches", patches.shape
    patches = normalize(patches)
    pca = PCA(n_components=3, whiten=True)
    pca.fit(patches)
    # Getting the codebook
    codebook, distortion = vq.kmeans(patches, K)
    print "codebook", codebook.shape
    ### Visualizing the codebook clustering
    #assignment, cdist = vq.vq(patches, codebook)
    #plt.scatter(patches[:, 0], patches[:,1], c=assignment)
    #plt.show()
    ########################
    histTrain = computeHistogram(patches, size, codebook)
    patchesTest, sizeTest = view_patches(X_test, P)
    patchesTest = normalize(patchesTest)
    pca.fit(patchesTest)
    assert(size == sizeTest)
    histTest = computeHistogram(patchesTest, sizeTest, codebook)

    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),\
        n_estimators=600, learning_rate=0.01, algorithm="SAMME")
    start_time = time()
    bdt_discrete.fit(histTrain, Y_train)
    t = time()-start_time
    print("\nTrain set score: %f\n" %bdt_discrete.score(histTrain, Y_train))
    print "Time fit:", t
    print("\nTest set score: %f\n" %bdt_discrete.score(histTest, Y_test))

def computeHistogram(patches, size, codebook):
    """
    Computing histogram for each image based on the codebook
    """
    print "Computing histograms ...."
    # Compute histograms
    #descriptors = np.arange(len(X_train))
    histograms = []
    for i in range(0, len(patches), size):
        #### need to get patches here as well
        p = patches[i:i+size]
        code, dist = vq.vq(p, codebook)
        #plt.scatter(p[:,0], p[:,1], c=code)
        #plt.show()
        histograms_of_words, bin_edges = histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
        #plt.hist(code, bins=range(codebook.shape[0] + 1), normed=True)
        #plt.title("Histogram of visual words")
        #plt.show()
        histograms.append(histograms_of_words)
    histograms = np.array(histograms)
    print "histograms", histograms.shape
    return histograms
def view_patches(images, patch_size):
    """
    Getting patches of a specific size from an array of images
    """
    print "Getting patches..."
    n, h, w = images.shape
    esize = images.itemsize

    nx = w-patch_size+1
    ny = h-patch_size+1

    patches = as_strided(images, shape=[n,ny,nx,patch_size,patch_size],
                         strides=[h*w*esize, w*esize, esize, w*esize, esize])
    print "Shape of patches prereshape", patches.shape
    p = np.reshape(patches, (n*ny*nx, patch_size*patch_size))
    #patches = np.reshape(n*nx*ny, patch_size*patch_size)

    return p, ny*nx
def CNNplot(X, Y, K, D, option, min_dim):
    X = X.reshape(X.shape[0], min_dim, min_dim, 1)
    Y = np_utils.to_categorical(Y, 3)
    model = Sequential()
    if option == 1:
        F = 5
        B = 32
    if option == 2:
        F = 11
        B = 32
    if option == 0:
        F = 3
        B = 10
    if option == 4:
        F = 3
        B = 32
    model.add(Convolution2D(K, (F, F),kernel_initializer='random_normal',
     activation='relu', input_shape=(min_dim,min_dim, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(BatchNormalization())
    if option >=1:
        model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(D, activation='relu',kernel_regularizer=regularizers.l2(0.01),\
activity_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization())

    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))

    #  Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    #  Fit model on training data
    t0 = time()
    history = model.fit(X, Y, validation_split=0.2,
              batch_size=B, nb_epoch=8, verbose=1)
    print "Time: ", time()-t0
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print "________________________________________"


    # Getting the weights of the first Convolution layer to display
    """
    conv_layer = model.layers[1]
    kernels = conv_layer.get_weights()[0]
    print "kernels array has shape", kernels.shape
    kernel_h, kernel_w, num_input_channels, num_kernels = kernels.shape

    # Parameters for display
    KERNELS_PER_ROW = 8 # We will display an image with 8 kernels per row
    MARGIN = 2 # Spacing between kernels
    ZOOM = 8 # Magnification factor for kernels

    # Determine # of rows
    num_rows = int(np.ceil(float(num_kernels) / KERNELS_PER_ROW))

    # Determine dimensions of display
    h = (kernel_h + MARGIN)*num_rows + MARGIN
    w = (kernel_w + MARGIN)*KERNELS_PER_ROW + MARGIN

    # Allocate space for display
    display = 0.9*np.ones((h, w), dtype=kernels.dtype)

    # Get kernel min/max for rescaling below
    kmin = kernels.min()
    kmax = kernels.max()

    # For each kernel
    for i in range(num_kernels):

        # Find out row/column in display image
        row = i / KERNELS_PER_ROW
        col = i % KERNELS_PER_ROW

        # Get pixel coordinates
        y0 = row*(kernel_h + MARGIN) + MARGIN
        x0 = col*(kernel_w + MARGIN) + MARGIN

        # Get the i'th kernel
        ki = kernels[:,:,0,i]

        # Rescale to [0,1] interval
        ki = (ki - kmin) / (kmax - kmin)

        # Blit it into the display image
        display[y0:y0+kernel_h, x0:x0+kernel_w] = ki

    # Magnify the display image
    display = cv2.resize(display, (ZOOM*w, ZOOM*h),
                         interpolation=cv2.INTER_NEAREST)

    # Show it
    cv2.imshow('Kernels', display)
    cv2.waitKey(0)
    """


if __name__=='__main__':
    main()
sys.exit()
