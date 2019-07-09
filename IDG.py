import numpy as np
import keras
import cv2

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, df, batch_size=32, dim=(155,220), n_channels=3, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.labels = df["label"]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.df.shape[0] / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        rows = [self.df.iloc[k] for k in indexes]
        X, y = self.__data_generation(rows)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.df.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, rows):
        x_1 = np.empty((self.batch_size, *self.dim, self.n_channels))
        x_2 = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        for i in range(len(rows)):
            image_1 = cv2.imread(rows[i]["image_1"])
            image_1 = cv2.resize(image_1,(220,155))
            image_1 = cv2.bitwise_not(image_1)
            image_1=np.array(image_1)
            image_2 = cv2.imread(rows[i]["image_2"])
            image_2 = cv2.resize(image_2,(220,155))
            image_2 = cv2.bitwise_not(image_2)
            image_2=np.array(image_2)
            '''mean_center_1 = image_1 - np.mean(image_1, axis = None)
            mean_center_2 = image_2 - np.mean(image_2, axis = None)
            
            std_1 = np.std(image_1)
            std_2 = np.std(image_2)
            if(std_1 == 0 or std_1 == np.nan):
                std_1 = 1
            if(std_2 == 0 or std_2 == np.nan):
                std_2 = 1
            standardized_img_1 = image_1/std_1
            standardized_img_2 = image_2/std_2
            '''
            x_1[i,] = image_1/255
            x_2[i,] = image_2/255
            y[i] = rows[i]["label"]


        return [x_1, x_2], y