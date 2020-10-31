from __future__ import print_function, division
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
import matplotlib.pyplot as plt
import pydicom as dicom
import os
import shutil
import cv2
from keras.preprocessing.image import load_img
import warnings
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
save_fit = False
model_save_loc = "saved_model"

main_data = "Patient and Treatment Characteristics.csv"
sec_data = "HNSCC Clinical Data.csv"
test_file = "test_2.csv"
target_variable = "PostRT Skeletal Muscle status"

# if true, converted images will be in png format instead of jpg
png = False

# list of names for duplicate columns
var_blacklist = ["Gender","Age at Diag"]

# folder containing Cancer Imagery
load_dir = "Cancer Imagery"

# directory to save data such as converted images
save_dir = "D:\\Cancer_Project\\converted_img"

# number of images to use from each patient
num_patient_imgs = 1

#if true, converted dicom images will be deleted after use
del_converted_imgs = False

def collect_img_dirs(data_folder):
    img_directories = []

    datas = os.listdir(data_folder)[0]
    Patients = os.listdir(data_folder + "\\" + datas)

    for patients in Patients:
        data = os.listdir(data_folder + "\\" + datas + "\\" + patients)[0]
        data2 = os.listdir(data_folder + "\\" + datas + "\\" + patients + "\\" + data)[0]
        img_directories.append(data_folder + "\\" + datas + "\\" + patients + "\\" + data + "\\" + data2)

    return img_directories

load_dirs = collect_img_dirs(load_dir)

num_epochs = 80

def convert_img(png_boolean,dcm_folder_path,save_path, num_imgs_patient):
    png = png_boolean

    # get elements in dicom path as list
    images_path = os.listdir(dcm_folder_path)[:num_imgs_patient]
    print(dcm_folder_path)
    print(images_path)

    for n, image in enumerate(images_path):
        ds = dicom.dcmread(os.path.join(dcm_folder_path,image),force=True)
        pixel_array_numpy = ds.pixel_array

        if png == False:
            image = image.replace(".dcm",".jpg")
        elif png == True:
            image = image.replace(".dcm",".png")
        cv2.imwrite(os.path.join(save_path,ds.PatientID+"_"+image),pixel_array_numpy)

#for dirs in load_dirs:
#    convert_img(png, dirs,save_dir, num_patient_imgs)

def combine_data(data_file_1,data_file_2):
    file_1 = pd.read_csv(data_file_1)
    file_2 = pd.read_csv(data_file_2)
    common_ids = []

    ids_1 = file_1.iloc[:,0]
    ids_2 = file_2.iloc[:,0]

    # determine the largest dataset to put first in the for statement
    if ids_1.shape[0] > ids_2.shape[0]:
        longest_ids = ids_1.values.tolist()
        shortest_ids = ids_2.values.tolist()
    elif ids_1.shape[0] < ids_2.shape[0]:
        longest_ids = ids_2.values.tolist()
        shortest_ids = ids_1.values.tolist()

    for i in longest_ids:
        for z in shortest_ids:
            if i == z:
                common_ids.append(i)

    adapted_1 = file_1.iloc[common_ids]
    adapted_2 = file_2.iloc[common_ids]
    combined_dataset = adapted_1.join(adapted_2)

    for i in var_blacklist:
        combined_dataset = combined_dataset.drop(i,axis=1)

    return combined_dataset

main_data = combine_data(main_data,sec_data)


def model(data_file,test_file,target_variable,epochs_num):

    def format_data(data_file, test_file, target_var):

        if str(type(data_file)) == "<class 'pandas.core.frame.DataFrame'>":
            df = data_file
        elif main_data[-4:] == ".csv":
            df = pd.read_csv(data_file)

        #Recognizing what variables are in the input data
        input_data = pd.read_csv(test_file)
        input_vars = input_data.columns.tolist()

        #collect data for the variables from main dataset
        dataset = df[input_vars]


        # Append y data for target column into new dataset
        y_data = df[target_var]
        dataset = dataset.assign(target_variable=y_data)
        target_name = str(target_var)
        dataset.rename(columns={'target_variable':target_name},inplace=True)

        return dataset

    adapted_dataset = format_data(data_file, test_file,target_variable)


    def NN(data_file, target_var, epochs_num):

        # Get data. Data must already be in a Pandas Dataframe
        df = data_file

        #y data
        labels = df[target_var]
        #x data
        features = df.drop(columns=[target_var])

        X = features
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model = Sequential([tf.keras.layers.Flatten()])
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.8),
                      metrics=['accuracy'])

        fit = model.fit(X_train, y_train, epochs=epochs_num, batch_size=5)

        # plotting
        history = fit

        def plot(model_history,metric,graph_title):
            history = model_history
            plt.plot(history.history[metric])
            plt.title(graph_title)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.show()

        plot(history,'accuracy','model accuracy')
        plot(history,'loss','model loss')

        def save_fitted_model(model,save_location):
            model.save(save_location)

        if save_fit == True:
            save_fitted_model(model,model_save_loc)

        print(model.predict(X_test, batch_size=1))

    NN(adapted_dataset,target_variable,epochs_num)

#model(main_data,test_file,target_variable,num_epochs)

def image_model(save_loc,data_file,test_file,target_var):

    def format_data(data_file, test_file, target_var):

        if str(type(data_file)) == "<class 'pandas.core.frame.DataFrame'>":
            df = data_file
        elif main_data[-4:] == ".csv":
            df = pd.read_csv(data_file)

        #Recognizing what variables are in the input data
        input_data = pd.read_csv(test_file)
        input_vars = input_data.columns.tolist()

        #collect data for the variables from main dataset
        dataset = df[input_vars]

        # Append y data for target column into new dataset
        y_data = df[target_var]
        dataset = dataset.assign(target_variable=y_data)
        target_name = str(target_var)
        dataset.rename(columns={'target_variable':target_name},inplace=True)

        return dataset

    adapted_dataset = format_data(data_file, test_file,target_var)
    adapted_dataset.index.names = ["ID"]

    img_array = np.array([])
    matching_ids = []
    img_list = os.listdir(save_loc)
    for imgs in img_list:
        for ids in adapted_dataset.index:
            ids = int(ids)
            if ids == int(imgs[9:13]):
                matching_ids.append(ids)
                matching_ids = list(dict.fromkeys(matching_ids))

                for ids in matching_ids:
                    if ids == int(imgs[9:13]):
                        img = load_img(os.path.join(save_loc, imgs))
                        img_numpy_array = img_to_array(img)
                        img_array = np.append(img_array,img_numpy_array)

    img_array = np.reshape(img_array,(len(matching_ids),786432))

    adapted_dataset = adapted_dataset.loc[matching_ids]

    def model(pd_data,input_imagery,target_var):

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Clinical
        # Get data
        df = pd_data

        # y data
        labels = df[target_var]
        # x data
        features = df.drop(columns=[target_var])

        X = features
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        scaler = StandardScaler().fit(X_train)
        X_train_clinical = scaler.transform(X_train)
        X_test_clinical = scaler.transform(X_test)

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        X_train = X_train.to_numpy()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Image

        img_data = input_imagery

        X_train_img, X_test_img = train_test_split(input_imagery,test_size=0.2,random_state=42)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        data = np.concatenate((X_train_img,X_train),axis=1)
        data_test = np.concatenate((X_test,X_test_img),axis=1)

        model = Sequential()
        model.add(Dense(12,input_shape=[786456,],activation="relu"))
        model.add(Dense(8,activation="relu"))
        model.add(Dense(1,activation="sigmoid"))

        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["accuracy"])

        model.fit(data,y_train,epochs=50,batch_size=5)

        print(model.predict(data_test))
        print(y_test)


    model(adapted_dataset,img_array,target_var)

image_model(save_dir,main_data,test_file,target_variable)

# delete converted dicom images after use if boolean is true
if del_converted_imgs == True:
    folder = save_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

