from matplotlib import pyplot as plt
from skimage.feature import hog
from matplotlib import patches
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from joblib import dump, load
from pull_digit import labeled_user_image
import cv2

model_path = 'digit_knn_model.sav'

def fetch_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    x = np.array(mnist.data, 'int16')
    y = np.array(mnist.target, 'int')
    return x, y

def extract_hog_features(data):
    list_hog_fd = []
    for feature in data:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        list_hog_fd.append(fd)
    return np.array(list_hog_fd, 'float64')

def train_model(x_train, y_train, model_path):
    x_train_hog = extract_hog_features(x_train)
    knn = KNeighborsClassifier()
    knn.fit(x_train_hog, y_train)
    dump(knn, model_path)

def knn_score(x_test, y_test, model_path):
    knn = load(model_path)
    x_test_hog = extract_hog_features(x_test)
    score = knn.score(x_test_hog, y_test)
    print(f"score is: {np.round(score * 100, 2)}%")
    return x_test_hog, y_test

def knn_predict(image, model_path):
    knn = load(model_path)
    return knn.predict(image)[0]

def read_image(rects, nums):
    xx = []
    yy = []
    for n in range(len(nums)):
        x = rects[n][0][0]
        y = rects[n][0][1]
        xx.append(x)
        yy.append(y)

    max_x = np.max(xx)
    max_y = np.max(yy)

    digits = np.zeros((max_x + 5, max_y + 5), object)

    return digits

def get_string(arr):
    dd = []
    for x in range(arr.shape[0] - 1):
        for y in range(arr.shape[1] - 1):
            if type(arr[x][y]) == list:
                dd.append(arr[x][y][0])

    print("the numers in the image are :")
    print(', '.join(str(x) for x in dd))

def new(image, padd):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    nums, ph, rects = labeled_user_image(image, padd)
    digits = read_image(rects, nums)
    for n in range(len(nums)):
        rect = patches.Rectangle(rects[n][0], rects[n][1], rects[n][2], linewidth=1, edgecolor='g',
                                 facecolor='none')

        ax.add_patch(rect)
        ex = knn_predict(nums[n], model_path)

        digits[rects[n][0][0]][rects[n][0][1]] = [ex]
        ax.text(rects[n][0][0] + 3, rects[n][0]
                [1] - 3, str(int(ex)), style='italic')

        plt.axis("off")

    get_string(digits)
    plt.show()


def view(image, pad):
    nums, ph, rects = labeled_user_image(image, pad)
    plt.figure()
    for n in range(len(ph)):
        plt.subplot(8, 11, n + 1)
        plt.imshow(ph[n], "gray")
        ex = knn_predict(nums[n], model_path)
        title_obj = plt.title(str(ex))
        plt.setp(title_obj, color='r')
        plt.axis("off")
    plt.show()

def main():
    x, y = fetch_mnist_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    
    train_model(x_train, y_train, model_path)
    knn_score(x_test, y_test, model_path)

if __name__ == '__main__':
    main()
