import cv2
import numpy as np
import os
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ImageClassifier:
    def __init__(self, root_dir, num_clusters=50):
        self.root_dir = root_dir
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.svm = svm.SVC(kernel='linear', probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.sift = cv2.SIFT_create()

    def load_images(self, directory):
        images = []
        labels = []
        class_labels = {'yes': 0, 'no': 1}
        for label, class_id in class_labels.items():
            class_path = os.path.join(directory, label)
            for filename in os.listdir(class_path):
                filepath = os.path.join(class_path, filename)
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(image)
                    labels.append(class_id)
        return images, labels

    def extract_features(self, images):
        all_descriptors = []
        for image in images:
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            if descriptors is not None:
                all_descriptors.extend(descriptors.astype(np.float32))
        return all_descriptors

    def train_kmeans(self, descriptors):
        self.kmeans.fit(np.array(descriptors, dtype=np.float32))

    def build_histograms(self, images):
        histograms = []
        for image in images:
            kp, des = self.sift.detectAndCompute(image, None)
            if des is not None:
                des = des.astype(np.float32)
                hist = np.zeros(self.num_clusters)
                predictions = self.kmeans.predict(des)
                for p in predictions:
                    hist[p] += 1
                histograms.append(hist)
            else:
                histograms.append(np.zeros(self.num_clusters))
        return self.scaler.fit_transform(histograms)

    def train(self, histograms, labels):
        self.svm.fit(histograms, labels)

    def evaluate(self, histograms, labels):
        predictions = self.svm.predict(histograms)
        print(classification_report(labels, predictions))

train_dir = 'C:\\Users\\niuka\\lab\\ass\\music\\dataset\\train'
test_dir = 'C:\\Users\\niuka\\lab\\ass\\music\\dataset\\test'  

classifier = ImageClassifier(train_dir, num_clusters=50)
train_images, train_labels = classifier.load_images(train_dir)
train_descriptors = classifier.extract_features(train_images)
classifier.train_kmeans(train_descriptors)
train_histograms = classifier.build_histograms(train_images)
train_hist, test_hist, train_labels, test_labels = train_test_split(train_histograms, train_labels, test_size=0.2, random_state=42)
classifier.train(train_hist, train_labels)


test_images, test_labels = classifier.load_images(test_dir)
test_descriptors = classifier.extract_features(test_images)
test_histograms = classifier.build_histograms(test_images)
classifier.evaluate(test_histograms, test_labels)