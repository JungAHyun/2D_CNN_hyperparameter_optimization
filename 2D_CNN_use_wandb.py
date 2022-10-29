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
config_wandb =0
num = 1

# wandb.init(
#     project="my-test-project",
#     config={
#         "conv_1": 32,
#         "activation_1": "swish",

#     #    "kernel_size": (3, 3),
#         "pool_size": (2, 2),
#     #    "dropout": 0.3,
#         "conv_2": 64,
#         "activation_out": "swish",
#         "optimizer": "adam",
#         "loss": "sparse_categorical_crossentropy",
#         "metric": "accuracy",
#         "epoch": 6,
#         "batch_size": 64
#     })

# config = wandb.config



# def make_wandb():
#     global config_wandb
#     # Default values for hyperparameters we're going to sweep over
#     config_defaults = {
#         "conv_1": 128,
#         "activation_1": "swish",
#         "kernel_size":((7, 7)),
#         "pool_size": (2, 2),
#         # "dropout": 0.1,
#         "conv_2": 128,
#         "conv_3": 128,
#         "optimizer": "adam",
#         "loss": "mae",
#         "metric": "mae",
#         "epoch": 50,
#         "batch_size": 64,
#         'learning_rate' : 0.0001

#     }

#     # Initialize a new wandb run
#     wandb.init(config=config_defaults, group='first_sweeps')
        
#     # config is a variable that holds and saves hyperparameters and inputs
#     config_wandb = wandb.config



# def make_2d_cnn(model):
#     global config_wandb
#     model.add(Conv2D(config_wandb.conv_1, config_wandb.kernel_size, padding='same', activation=config_wandb.activation_1, input_shape=(59,21,1)))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(config_wandb.conv_2, config_wandb.kernel_size, padding='same', activation=config_wandb.activation_1))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Conv2D(config_wandb.conv_3, config_wandb.kernel_size, padding='same', activation=config_wandb.activation_1))
#     model.add(MaxPooling2D(2, 2))
#     model.add(Flatten())
#     model.add(Dense(2048, activation=config_wandb.activation_1))
#     model.add(Dense(2048, activation=config_wandb.activation_1))
#     model.add(Dense(2048, activation=config_wandb.activation_1))
#     model.add(Dense(2048, activation=config_wandb.activation_1))
#     model.add(Dense(1, activation=None))

#     return model  


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for i in range(len(gpus)):
        tf.config.experimental.set_memory_growth(gpus[i], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


angle_list = None
pp_list = None

print("loading data...")

for i in range(1,13):
    input_path= 'real_input_data\pp_to_ja'+str(i)+'.csv'       #족저압 input데이터  파일경로
    #input 데이터 csv파일 불러와서 데이터프레임 저장 
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

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
            print('.', end='')


def train():
    # global config_wandb

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


    save_filepath = '2022_10_cnn_result\\case2_result.csv'
    f =  open(save_filepath, 'w', encoding='utf-8', newline="")
    wr = csv.writer(f)
    for i in range(len(y_test)-1):
        wr.writerow([y_test[i], y_pred[i]])

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
    global num
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
        epoch= 50,
        batch_size= 64,
        learning_rate = 0.0001
        )

    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    # wandb.run.name = 'Case' + str(num)
    # num=num+1
    # print(num)





if __name__ == "__main__":

    angle = angle.squeeze()
    pp = pp.reshape(-1, 59,21,1)

    print(angle.shape)      #(5896, )
    print(pp.shape)          #(5896, 59,21)

    x_train, x_test, y_train, y_test = train_test_split(pp, angle, test_size=0.2, shuffle=True)    






    init_wandb()
    train()

    

  

    # sweep_id = get_sweep_id()
    # wandb.agent(sweep_id,train())


    # y_pred = model.predict(x_test)
    # y_pred = y_pred.squeeze()
    

    #코드
    # wandb.log({
    #     "r2 score": r2_score(y_true=y_test, y_pred=y_pred)
    # })


    # save_filepath = '2022_10_cnn_result\\case2_result.csv'
    # f =  open(save_filepath, 'w', encoding='utf-8', newline="")
    # wr = csv.writer(f)
    # for i in range(len(y_test)-1):
    #     wr.writerow([y_test[i], y_pred[i]])

    # print("========= Result =========")
    # print("MAE: ",mean_absolute_error(y_true=y_test, y_pred=y_pred))
    # print("MAE2: ", np.mean(np.abs(y_test - y_pred)))
    # print("std: ", np.std(np.absolute(np.subtract(y_test, y_pred))))
    # print("=========================")
    # print("ME: ", np.mean(np.subtract(y_test, y_pred)))
    # print("std: ", np.std(np.subtract(y_test, y_pred)))
    # print("=========================")
    # print("mean relative error: ", np.mean(np.divide(np.absolute(np.subtract(y_test, y_pred)), y_test))*100)
    # print("=========================")        
    # print("r2 score: ", r2_score(y_true=y_test, y_pred=y_pred))
    # print("=========================")
    # model.summary()
    
 
# # patience 매개변수는 성능 향상을 체크할 에포크 횟수입니다
#     early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#     history = model.fit(pp, angle, epochs=EPOCHS,
#                     validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

#     plot_history(history)
    



    