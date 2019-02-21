import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from numpy import sqrt, histogram
np.random.seed(123)
from subprocess import check_output
print(check_output(["ls", "input"]).decode("utf8"))

TRAIN_DATA = "input/train"
TEST_DATA = "input/test"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[11:-4] for s in test_files])
print "test ids", len(test_ids), test_ids[:3]
print "type 1", len(type_1_ids)
print "type 2", len(type_2_ids)
print "type 3", len(type_3_ids)
print "test", len(test_ids)
def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type
    """
    if image_type == "Type_1" or \
        image_type == "Type_2" or \
        image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or \
          image_type == "AType_2" or \
          image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def maxHist(hist):
    """
    Algorithm for getting the maximum histogram of a given patch
    """
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i-position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif(height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area =  maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    """
    Find the maximum connected rectangle region in an image
    """
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r-1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (int(maxArea[3]+1-maxArea[0]/abs(maxArea[1]-maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])

def cropCircle(img):
    """
    Cropping the circle often seen in cervigrams using maxRect and maxHist to detect
    a maximum connected region
    """
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1]*256/img.shape[0]),256)
    else:
        tile_size = (256, int(img.shape[0]*256/img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key = cv2.contourArea, reverse = True)[0]

    ff = np.zeros((gray.shape[0],gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0]+2,gray.shape[1]+2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 1)
    #cv2.circle(ff, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 3, 3, -1)

    rect = maxRect(ff)
    img_crop = img[min(rect[0],rect[2]):max(rect[0],rect[2]), min(rect[1],rect[3]):max(rect[1],rect[3])]
    cv2.rectangle(ff,(min(rect[1],rect[3]),min(rect[0],rect[2])),(max(rect[1],rect[3]),max(rect[0],rect[2])),3,2)

    #plt.subplot(121)
    #plt.imshow(img)
    #plt.subplot(122)
    #plt.imshow(ff)
    #plt.show()

    return img_crop
def Ra_space(img, Ra_ratio, a_threshold):
    """
    Assigning each pixel in the image 2 values:
    a = max(value of channel, a_threshold) (a_threshold = 150)
    R = distance from center pixel
    """
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)

    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])

    return Ra

def getData():
    """
    Return X (a numpy array of RGB images of segmented cervigrams with same
    size 58x58), Y (the labels of the images), and the size of the images
    """
    k = 0
    ids = [sorted(type_1_ids), sorted(type_2_ids), sorted(type_3_ids)]
    X_train = []
    Y_train = []
    min_dim = 1000
    while k < 3:
        for image_id in ids[k]:
            img = get_image_data(image_id, 'Type_%i' % (k+1))
            img = cropCircle(img)
            w = img.shape[0]
            h = img.shape[1]
            imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
            # Saturating the a-channel at 150 helps avoiding wrong segmentation
            # in the case of close-up cervix pictures where the bloody os is falsly segemented as the cervix.
            Ra = Ra_space(img, 1.0, 150)
            a_channel = np.reshape(Ra[:,1], (w,h))
            plt.subplot(121)
            plt.imshow(a_channel)

            g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
            image_array_sample = shuffle(Ra, random_state=0)[:1000]
            g.fit(image_array_sample)
            labels = g.predict(Ra)
            labels += 1 # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.

            # The cluster that has the highest a-mean is selected.
            labels_2D = np.reshape(labels, (w,h))
            gg_labels_regions = measure.regionprops(labels_2D, intensity_image = a_channel)
            gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
            cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

            mask = np.zeros((w * h,1),'uint8')
            mask[labels==cervix_cluster] = 255
            mask_2D = np.reshape(mask, (w,h))

            cc_labels = measure.label(mask_2D, background=0)
            regions = measure.regionprops(cc_labels)
            areas = [prop.area for prop in regions]

            regions_label = [prop.label for prop in regions]
            largestCC_label = regions_label[areas.index(max(areas))]
            mask_largestCC = np.zeros((w,h),'uint8')
            mask_largestCC[cc_labels==largestCC_label] = 255

            img_masked = img.copy()
            img_masked[mask_largestCC==0] = (0,0,0)
            img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);

            _,thresh_mask = cv2.threshold(img_masked_gray,0,255,0)

            kernel = np.ones((11,11), np.uint8)
            thresh_mask = cv2.dilate(thresh_mask, kernel, iterations = 1)
            thresh_mask = cv2.erode(thresh_mask, kernel, iterations = 2)
            _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

            main_contour = sorted(contours_mask, key = cv2.contourArea, reverse = True)[0]

            x,y,w,h = cv2.boundingRect(main_contour)
            cv2.rectangle(img,(x,y),(x+w,y+h),255,3)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sub_img = img_gray[y:y+h, x:x+w]
            dim = max(h, w)
            if dim < min_dim:
                min_dim = dim
            sub_img = cv2.resize(sub_img, (dim, dim))
            X_train.append(sub_img)
            Y_train.append(k+1)
            #print "shape sub_img", sub_img.shape
            #cv2.imshow("sub img", sub_img)
            #cv2.waitKey(0)
            #plt.subplot(122)
            #plt.imshow(sub_img)
            #plt.show()

        k = k + 1;

    # Resizing all the images to min_dim x min_dim
    #print "all Y", Y_train
    new_X_train = []
    for img in X_train:
        new_img = cv2.resize(img, (min_dim, min_dim))
        new_X_train.append(new_img)
        #cv2.imshow("train img", img)
        #cv2.waitKey(0)

    # Creating X a np array of all the rgb images
    print "X_train original", len(new_X_train), new_X_train[0].shape
    X = np.stack(tuple(new_X_train))


    # - normalize each uint8 image to the value interval [0, 1] as float image
    X = X * (2.0/255.0) - 1.0


    # Use CNN with F = 5, K = 50, P = 0, S = 1
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import Normalizer


    Y = LabelEncoder().fit_transform(Y_train).reshape(-1)
    print "After getData X: ", X.shape, "Y:", Y.shape
    return X, Y, min_dim


def getSubmitData():
    """
    Returning X (an array of RGB images of segmented cervigrams with the same size)
    and the size of the images
    """
    X_submit = []
    min_dim = 1000
    for image_id in test_ids:

        img = get_image_data(image_id, 'Test')
        img = cropCircle(img)
        w = img.shape[0]
        h = img.shape[1]

        imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);

        # Saturating the a-channel at 150 helps avoiding wrong segmentation
        # in the case of close-up cervix pictures where the bloody os is falsly segemented as the cervix.
        Ra = Ra_space(img, 1.0, 150)
        a_channel = np.reshape(Ra[:,1], (w,h))
        plt.subplot(121)
        plt.imshow(a_channel)

        g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
        image_array_sample = shuffle(Ra, random_state=0)[:1000]
        g.fit(image_array_sample)
        labels = g.predict(Ra)
        labels += 1 # Add 1 to avoid labeling as 0 since regionprops ignores the 0-label.

        # The cluster that has the highest a-mean is selected.
        labels_2D = np.reshape(labels, (w,h))
        gg_labels_regions = measure.regionprops(labels_2D, intensity_image = a_channel)
        gg_intensity = [prop.mean_intensity for prop in gg_labels_regions]
        cervix_cluster = gg_intensity.index(max(gg_intensity)) + 1

        mask = np.zeros((w * h,1),'uint8')
        mask[labels==cervix_cluster] = 255
        mask_2D = np.reshape(mask, (w,h))

        cc_labels = measure.label(mask_2D, background=0)
        regions = measure.regionprops(cc_labels)
        areas = [prop.area for prop in regions]

        regions_label = [prop.label for prop in regions]
        largestCC_label = regions_label[areas.index(max(areas))]
        mask_largestCC = np.zeros((w,h),'uint8')
        mask_largestCC[cc_labels==largestCC_label] = 255

        img_masked = img.copy()
        img_masked[mask_largestCC==0] = (0,0,0)
        img_masked_gray = cv2.cvtColor(img_masked, cv2.COLOR_RGB2GRAY);

        _,thresh_mask = cv2.threshold(img_masked_gray,0,255,0)

        kernel = np.ones((11,11), np.uint8)
        thresh_mask = cv2.dilate(thresh_mask, kernel, iterations = 1)
        thresh_mask = cv2.erode(thresh_mask, kernel, iterations = 2)
        _, contours_mask, _ = cv2.findContours(thresh_mask.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        main_contour = sorted(contours_mask, key = cv2.contourArea, reverse = True)[0]

        x,y,w,h = cv2.boundingRect(main_contour)
        cv2.rectangle(img,(x,y),(x+w,y+h),255,3)
        sub_img = img_masked[y:y+h, x:x+w]
        dim = max(h, w)
        if dim < min_dim:
            min_dim = dim
        sub_img = cv2.resize(sub_img, (dim, dim))
        X_submit.append(sub_img)

    new_X_submit = []
    for img in X_submit:
        new_img = cv2.resize(img, (min_dim, min_dim))
        new_X_submit.append(new_img)
        #cv2.imshow("train img", img)
        #cv2.waitKey(0)

    # Creating X a np array of all the rgb images

    X = np.stack(tuple(new_X_submit))


    # - normalize each uint8 image to the value interval [0, 1] as float image
    # - rgb to gray
    # - downsample image to rescaled_dim X rescaled_dim
    # - L2 norm of each sample = 1

    classifier_input = X.reshape(-1, min_dim*min_dim).astype(np.float32)
    X = X * (2.0/255.0) - 1.0
    print "X (to be submitted)", X.shape
    return X, min_dim
