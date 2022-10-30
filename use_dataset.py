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
   # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


x_train_path = 'x_train_dataset.csv'
x_test_path = 'x_test_dataset.csv'
y_train_path = 'y_train_dataset.csv'
y_test_path = 'y_test_dataset.csv'

def get_dataset():
    x_train_df=pd.read_csv(x_train_path, header=None)
    y_train_df=pd.read_csv(y_train_path, header=None)
    x_test_df=pd.read_csv(x_test_path, header=None)
    y_test_df=pd.read_csv(y_test_path, header=None)

    x_train = x_train_df.iloc[:,:].to_numpy()
    y_train = y_train_df.iloc[:,:].to_numpy()
    x_test = x_test_df.iloc[:,:].to_numpy()
    y_test = y_test_df.iloc[:,:].to_numpy()

    return x_train, x_test, y_train, y_test


# y(관절각도) 데이터 가공하는 함수
def get_y_data(y_data):
    y_data_list = []
    for i in range(len(y_data[0])):
        tmp =  y_data[0][i]
        y_data_list.append(tmp)
    
    return np.array(y_data_list)


# x(족저압) 데이터 가공하는 함수
def get_x_data(x_data):
    x_data_list =[]
    for i in range(len(x_data)):
        data = x_data[i]
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
        x_data_list.append(converted_data)
    
    return np.array(x_data_list)



class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
            print('.', end='')


def train(x_train, x_test, y_train, y_test):
    # config is a variable that holds and saves hyperparameters and inputs
    config_wandb = wandb.config

    model = tf.keras.Sequential()
    model.add(Conv2D(config_wandb.conv, kernel_size =(7, 7), padding='same', activation=config_wandb.activation, input_shape=(59,21,1)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(config_wandb.conv, kernel_size =(5, 5), padding='same', activation=config_wandb.activation))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(config_wandb.conv, kernel_size =(3, 3), padding='same', activation=config_wandb.activation))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(2048, activation=config_wandb.activation))
    model.add(Dense(2048, activation=config_wandb.activation))
    model.add(Dense(2048, activation=config_wandb.activation))
    model.add(Dense(2048, activation=config_wandb.activation))
    model.add(Dense(1, activation=None))
    adam= tf.keras.optimizers.Adam(lr=config_wandb.learning_rate)

    model.compile(optimizer=adam,
              loss='mse',
              metrics=['mae'])
              
    model.fit(x_train, y_train, epochs=config_wandb.epoch, shuffle=True, batch_size = config_wandb.batch_size)


    y_pred = model.predict(x_test)
    y_pred = y_pred.squeeze()

    wandb.log({
        "r2 score": r2_score(y_true=y_test, y_pred=y_pred),
        "MAE": mean_absolute_error(y_true=y_test, y_pred=y_pred),
        "MRE": np.mean(np.divide(np.absolute(np.subtract(y_test, y_pred)), y_test))*100
    })


    # save_filepath = '2022_10_cnn_result\\case2_result.csv'
    # f =  open(save_filepath, 'w', encoding='utf-8', newline="")
    # wr = csv.writer(f)
    # for i in range(len(y_test)-1):
    #     wr.writerow([y_test[i], y_pred[i]])

    print("========= Result =========")
    print("MAE: ",mean_absolute_error(y_true=y_test, y_pred=y_pred))
    print("MAE2: ", np.mean(np.abs(y_test - y_pred)))
    print("std: ", np.std(np.absolute(np.subtract(y_test, y_pred))))
    print("=========================")
    print("ME: ", np.mean(np.subtract(y_test, y_pred)))
    print("std: ", np.std(np.subtract(y_test, y_pred)))
    print("=========================")
    print("mean relative error: ", np.mean(np.divide(np.absolute(np.subtract(y_test, y_pred)), y_test))*100)
    print("=========================")        
    print("r2 score: ", r2_score(y_true=y_test, y_pred=y_pred))
    print("=========================")
    model.summary()



def init_wandb():
    # Default values for hyperparameters we're going to sweep over
    # Set up your default hyperparameters before wandb.init
    # so they get properly set in the sweep
    hyperparameter_defaults = dict(
        conv =  128,
        activation = "swish",
        # kernel_size =((7, 7)),
        # pool_size = (2, 2),
        # "dropout": 0.1,
        optimizer ="adam",
        loss="mae",
        metric="mae",
        epoch= 1,
        batch_size= 64,
        learning_rate = 0.0001
        )

    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)




i = 1
if __name__ == "__main__":

    print("loading data...")
    x_train, x_test, y_train, y_test = get_dataset()

    print("processing data...")
    x_train = get_x_data(x_train)
    x_test =get_x_data(x_test)

    y_train = get_y_data(y_train)
    y_test =get_y_data(y_test)

    # print('x_train: ', x_train)
    # print('x_train len: ', len(x_train))
    # print('x_train[0]: ', x_train[0])
    # print('x_train[0] len: ', len(x_train[0]))
    # print('==============================')
    # print('y_train: ', y_train)
    # print('y_train len: ', len(y_train))
 
 
    
    x_train = x_train.reshape(-1, 59,21,1)
    x_test = x_test.reshape(-1, 59,21,1)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    print(x_train.shape)      #(144000, )
    print(x_test.shape)          #(144000, 59,21)
    print(y_train.shape)      #(144000, )
    print(y_test.shape)          #(144000, 59,21)
    

    init_wandb()

    train(x_train, x_test, y_train, y_test)


    

  
