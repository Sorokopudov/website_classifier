import os
import pickle
import time

import cv2
import numpy as np
from sklearn.metrics import recall_score, f1_score, accuracy_score
from tensorflow.python.keras.applications.mobilenet import preprocess_input
import tensorflow as tf

from app.image_search.bigan import BigBigan


class ImageClassifierAll:
    def __init__(self, model):
        self.all_skus = {}
        self.model = model
        self.predict_time = 0
        self.count_frame = 0
        self.top_k = 10
        self.time_search = 0

    def extract_features_from_img(self, cur_img, use_grayscale_img=False):
        img = cv2.resize(cur_img, self.model.get_input_size())
        if (use_grayscale_img):
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        feature = self.model.extract_feature(img)
        return feature

    def predict_top(self, img):
        from queue import PriorityQueue
        self.count_frame += 1
        before_time = time.time()
        target_features = self.extract_features_from_img(img)
        self.predict_time += time.time() - before_time
        before_time = time.time()
        customers = PriorityQueue()
        for dish, features_all in self.all_skus.items():
            for features in features_all:
                cur_distance = self.model.cosine_distance(target_features, features)
                # cur_distance = cur_distance[0][0]
                if cur_distance > 0.74:
                    # if cur_distance > 0.4:
                    customers.put((cur_distance, dish))

                if customers.qsize() > self.top_k:
                    customers.get()

        self.time_search += time.time() - before_time
        result = customers.queue
        result = sorted(result, key=lambda x: x[0])
        result = list(reversed(result))
        return result

    def predict(self, img, threshold):
        self.count_frame += 1
        before_time = time.time()
        target_features = self.extract_features_from_img(img)
        self.predict_time += time.time() - before_time
        max_distance = 0
        result_dish = 0

        for dish, features_all in self.all_skus.items():
            for features in features_all:
                cur_distance = self.model.cosine_distance(target_features, features)
                if cur_distance > max_distance:
                    max_distance = cur_distance
                    result_dish = dish
        if max_distance < threshold:
            return 0, max_distance

        return result_dish, max_distance
    def add_img(self, img_path, id_img, pickle_path):
        img = cv2.imread(img_path)
        cur_img = img
        feature = self.extract_features_from_img(cur_img)
        if id_img not in self.all_skus:
            self.all_skus[id_img] = []
        self.all_skus[id_img].append(feature)
        # pickle.dump(feature, open(pickle_path, "wb"))
        return feature

    def remove_by_id(self, id_img):
        if id_img in self.all_skus:
            self.all_skus.pop(id_img)

    def remove_all(self):
        self.all_skus.clear()

    def add_img_from_pickle(self, id_img, pickle_path):
        res = pickle.load(open(pickle_path, 'rb'))
        self.all_skus[id_img] = res

    def add_folder(self, folder_path, folder_name):
        files = os.listdir(folder_path)
        for file_name in files:
            self.add_img(os.path.join(folder_path, file_name), folder_name, "")

def create_model():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6  # dynamically grow the memory used on the GPU
    # alt_sess = tf.compat.v1.Session(graph=tf.Graph(), config=self.config)
    gan_sess = tf.compat.v1.Session(graph=tf.Graph(), config=config)
    return BigBigan(gan_sess, use_encoder=False, use_resnet=True)
if __name__=="__main__":

    classifier = ImageClassifierAll(create_model())
    classifier.add_folder("/home/sergej/DigitalDrill/gloves/data/dataset3classes_big/hands", "hands")
    classifier.add_folder("/home/sergej/DigitalDrill/gloves/data/dataset3classes_big/gloves", "gloves")
    classifier.add_folder("/home/sergej/DigitalDrill/gloves/data/dataset3classes_big/nothing", "no_hands")

    files = os.listdir("/home/sergej/DigitalDrill/gloves_dataset")
    tp = 0
    fp = 0
    fn = 0

    i = 0.74
    while i < 1:
        y_true = []
        y_pred = []
        for category in files:
            imgs = os.listdir(os.path.join("/home/sergej/DigitalDrill/gloves_dataset", category))
            for v in imgs:
                img = cv2.imread(os.path.join("/home/sergej/DigitalDrill/gloves_dataset", category, v))
                result_class, distance = classifier.predict(img, i)
                y_true.append(category)
                y_pred.append(result_class)

        print("Threshold = " + str(i))
        print(accuracy_score(y_true, y_pred))
        # print(recall_score(y_true, y_pred))
        print(f1_score(y_true, y_pred, average="macro"))
        print("---------")
        i += 0.01

