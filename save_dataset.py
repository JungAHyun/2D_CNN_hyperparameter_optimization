
import csv
from distutils.command.config import config
from unicodedata import name
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

x_train_filepath = 'x_train_dataset.csv'
y_train_filepath = 'y_train_dataset.csv'
x_test_filepath = 'x_test_dataset.csv'
y_test_filepath = 'y_test_dataset.csv'


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

# csv 생성 및 컬럼 작성
def make_csv():
        x_train_f=  open(x_train_filepath, 'w', encoding='utf-8', newline="")
        x_train_wr = csv.writer(x_train_f)
        
        return x_train_wr


def save_dataset(x_train, x_test, y_train, y_test):

    x_train_f=  open(x_train_filepath, 'w', encoding='utf-8', newline="")
    x_train_wr = csv.writer(x_train_f)
        
    x_test_f=  open(x_test_filepath, 'w', encoding='utf-8', newline="")
    x_test_wr = csv.writer(x_test_f)

    y_train_f=  open(y_train_filepath, 'w', encoding='utf-8', newline="")
    y_train_wr = csv.writer(y_train_f)

    y_test_f=  open(y_test_filepath, 'w', encoding='utf-8', newline="")
    y_test_wr = csv.writer(y_test_f)

    print(y_train[0]) 
    print(type(y_train[0]))  
    

    # x_train save 
    for i in range(len(x_train)):
        one_data= []
        for j in range(59):
            tmp = []
            for k in range(21):
                tmp.append((x_train[i][j][k][0]))
            one_data.append(str(tmp))
        x_train_wr.writerow(one_data)
    
    # x_test save 
    for i in range(len(x_test)):
        one_data= []
        for j in range(59):
            tmp = []
            for k in range(21):
                tmp.append((x_train[i][j][k][0]))
            one_data.append(str(tmp))
        x_test_wr.writerow(one_data)
    
    # y_train save 
    y_train_wr.writerow(y_train)

    # y_test save 
    y_test_wr.writerow(y_test)

    print('-------------------------- Save End --------------------------')
    print('x_train: ',len(x_train))
    print('x_test: ',len(x_test))
    print('y_train: ',len(y_train))
    print('y_test: ',len(y_test))

    




if __name__ == "__main__":

    angle, pp = get_csv()

    angle = angle.squeeze()
    pp = pp.reshape(-1, 59,21,1)
    print(pp)

    print(angle.shape)      #(144000, )
    print(pp.shape)          #(144000, 59,21)
    
    x_train, x_test, y_train, y_test = train_test_split(pp, angle, test_size=0.2, shuffle=True)  

    save_dataset(x_train, x_test, y_train, y_test)

