
import csv
from distutils.command.config import config
from unicodedata import name
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import wandb


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for i in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[i], True)
  except RuntimeError as e:
   
    print(e)



def get_csv():

    angle_list = None
    pp_list = None

    print("loading data...")

    for i in range(1,13):
        input_path= 'real_input_data\pp_to_ja'+str(i)+'.csv'      
        input_df=pd.read_csv(input_path, header=None)

        angle = input_df.iloc[:,1].to_numpy()
        pp = input_df.iloc[:, 3:].to_numpy()

        if pp_list is None:
            angle_list = angle
            pp_list = pp
        else:
            angle_list = np.concatenate((angle_list, angle), axis=0)
            pp_list = np.concatenate((pp_list, pp), axis=0)

    angle = angle_list
    pp = pp_list

    new_pp_list =[]

    print("processing data...")

    for i in range(len(pp)):
        data = pp[i]
        converted_data = []

        for line in data:
            tmp_line = line[1:-1].split(",")

            for i in range(len(tmp_line)):
                element = tmp_line[i]
                element = element.replace("\"", "")
                element = element.replace("\'", "")
                tmp_line[i] = element
                
            
            tmp_data = np.array(list(map(float,tmp_line)))
            converted_data.append(tmp_data)

        converted_data = np.array(converted_data)

        new_pp_list.append(converted_data)

    pp = np.array(new_pp_list)
    print("done")

    return angle, pp



def save_dataset(x_train, x_test, y_train, y_test):
    x_train_filepath = 'x_train_dataset.csv'
    y_train_filepath = 'y_train_dataset.csv'
    x_test_filepath = 'x_test_dataset.csv'
    y_test_filepath = 'y_test_dataset.csv'


    x_train.to_csv(x_train_filepath)
    y_train.to_csv(y_train_filepath)
    x_test.to_csv(x_test_filepath)
    y_test.to_csv(y_test_filepath)






if __name__ == "__main__":

    angle, pp = get_csv()

    angle = angle.squeeze()
    pp = pp.reshape(-1, 59,21,1)

    print(angle.shape)      #(5896, )
    print(pp.shape)          #(5896, 59,21)
    
    x_train, x_test, y_train, y_test = train_test_split(pp, angle, test_size=0.2, shuffle=True)  

    save_dataset(x_train, x_test, y_train, y_test)