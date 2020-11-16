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
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
save_fit = False
model_save_loc = "saved_model"

main_data = "HNSCC-HN1\\Copy of HEAD-NECK-RADIOMICS-HN1 Clinical data updated July 2020.csv"
sec_data = "HNSCC-3DCT\\MDPA Patient Data Final (Weight).csv"
test_file = "test_2.csv"
target_variable = "event_overall_survival"

# if true, converted images will be in png format instead of jpg
png = False

# list of names for duplicate columns
var_blacklist = ["Unnamed: 5"]

# folder containing Cancer Imagery
load_dir = "D:\Cancer_Project\\Cancer Imagery\\HEAD-NECK-RADIOMICS-HN1"

# directory to save data such as converted images
save_dir = "D:\\Cancer_Project\\converted_img"

# if true, attempt will be made to convert dicom files to jpg or png
convert_imgs = False

#if true, converted dicom images will be deleted after use
del_converted_imgs = False

# if true, image model will be ran instead of clinical only model
run_img_model = True

# if true, two data files will be expected for input
two_datasets = False

# if true, an additional file will be expected for testing
use_additional_test_file = False

# where image id is located in image names (start,end)
# only applies if using image model
img_id_name_loc = (2,6)

# Column of IDs in dataset. Acceptable values include "index" or a column name.
ID_dataset_col = "id"

# tuple with dimension of imagery. All images must equal this dimension
img_dimensions = (512, 512, 3)

# percentage value. Percentage of images that will be used.
# Prevents memory overload
image_model_timeout = 0.01

def collect_img_dirs(data_folder):
    img_directories = []

    for root, dirs, files, in os.walk(data_folder):
        for name in files:
            dir = os.path.join(root,name)
            img_directories.append(dir)

    return img_directories

if convert_imgs == True:
    load_dirs = collect_img_dirs(load_dir)

num_epochs = 120

def convert_img(png_boolean,dirs_list,save_path):
    png = png_boolean

    print("starting image conversion process")
    num_converted_img = 0
    for image in dirs_list:

        # filter out incompatible images
        if os.path.basename(image) != "1-1.dcm":
            ds = dicom.dcmread(image)
            pixel_array_numpy = ds.pixel_array

            if png == False:
                image = image.replace(".dcm",".jpg")
            elif png == True:
                image = image.replace(".dcm",".png")

            cv2.imwrite(os.path.join(save_path,ds.PatientID+"_"+os.path.basename(image)),pixel_array_numpy)

            ## Loading info
            num_imgs = len(dirs_list)
            num_converted_img = num_converted_img + 1
            percentage_done = (num_converted_img/num_imgs) * 100
            print(str(percentage_done) + " percent completed")

if convert_imgs == True:
    convert_img(png, load_dirs,save_dir)

def prep_data(data_file_1,data_file_2):
    file_1 = pd.read_csv(data_file_1)
    common_ids = []

    if ID_dataset_col != "index":
        file_1 = file_1.set_index(ID_dataset_col)

    ids_1 = file_1.index

    if two_datasets == True:
        file_2 = pd.read_csv(data_file_2)
        file_2 = file_2.set_index(ID_dataset_col)
        ids_2 = file_2.index
        # determine the largest dataset to put first in the for statement
        if ids_1.shape[0] > ids_2.shape[0]:
            longest_ids = ids_1.values.tolist()
            shortest_ids = ids_2.values.tolist()
        elif ids_1.shape[0] < ids_2.shape[0]:
            longest_ids = ids_2.values.tolist()
            shortest_ids = ids_1.values.tolist()
        elif ids_1.shape[0] == ids_2.shape[0]:
            longest_ids = ids_1.values.tolist()
            shortest_ids = ids_2.values.tolist()

        for i in longest_ids:
            for z in shortest_ids:
                if int(i) == int(z):
                    common_ids.append(i)

        adapted_1 = file_1.loc[common_ids]
        adapted_2 = file_2.loc[common_ids]
        combined_dataset = adapted_1.join(adapted_2)

        for i in var_blacklist:
            combined_dataset = combined_dataset.drop(i,axis=1)
        data = combined_dataset
    else:
        data = file_1

    return data

if two_datasets == True:
    main_data = prep_data(main_data,sec_data)
elif two_datasets == False:
    main_data = prep_data(main_data,None)

def model(data_file,test_file,target_variable,epochs_num):

    def format_data(data_file, test_file, target_var):

        if str(type(data_file)) == "<class 'pandas.core.frame.DataFrame'>":
            df = data_file
        elif main_data[-4:] == ".csv":
            df = pd.read_csv(data_file)

        if use_additional_test_file == True:
            #Recognizing what variables are in the test data
            input_data = pd.read_csv(test_file)
            input_vars = input_data.columns.tolist()

            #collect data for the variables from main dataset
            dataset = df[input_vars]

            # Append y data for target column into new dataset
            y_data = df[target_var]
            dataset = dataset.assign(target_variable=y_data)
            target_name = str(target_var)
            dataset = dataset.rename(columns={'target_variable':target_name},inplace=True)
        elif use_additional_test_file == False:
            dataset = df

        return dataset

    adapted_dataset = format_data(data_file, test_file,target_variable)

    # determine activation function (relu or tanh) from if there are negative numbers in target variable
    df_values = adapted_dataset.values
    df_values = df_values.flatten()
    for val in df_values:
        if val < 0:
            negative_vals = True
        else:
            negative_vals = False

    if negative_vals == True:
        act_func = "tanh"
    else:
        act_func = 'relu'

    def NN(data_file, target_var, epochs_num,activation_function):

        # Get data. Data must already be in a Pandas Dataframe
        df = data_file

        #y data
        labels = df[target_var]
        #x data
        features = df.drop(columns=[target_var])

        X = features
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model = Sequential([tf.keras.layers.Flatten()])
        model.add(Dense(24, activation=activation_function))
        model.add(Dense(24, activation=activation_function))
        model.add(Dense(24, activation=activation_function))
        model.add(Dense(10, activation=activation_function))
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
        print(y_test)

    NN(adapted_dataset,target_variable,epochs_num,act_func)

if run_img_model == False:
    model(main_data,test_file,target_variable,num_epochs)

def image_model(save_loc,data_file,test_file,target_var):
    print("starting image model")

    def format_data(data_file, test_file, target_var):

        if str(type(data_file)) == "<class 'pandas.core.frame.DataFrame'>":
            df = data_file
        elif main_data[-4:] == ".csv":
            df = pd.read_csv(data_file)

        if use_additional_test_file == True:
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
        elif use_additional_test_file == False:
            dataset = df

        return dataset

    adapted_dataset = format_data(data_file, test_file,target_var)
    adapted_dataset.index.names = ["ID"]

    img_array = np.array([])
    matching_ids = []
    img_list = os.listdir(save_loc)

    imgs_processed = 0

    print("starting data preparation process")
    for imgs in img_list:

        for ids in adapted_dataset.index:
            ids = int(ids)
            if ids == int(imgs[img_id_name_loc[0]:img_id_name_loc[1]]):
                matching_ids.append(ids)
                matching_ids = list(dict.fromkeys(matching_ids))

                total_img = len(img_list)
                timeout_val = int(total_img * image_model_timeout)
                if imgs_processed != timeout_val:
                    img = load_img(os.path.join(save_loc, imgs))
                    img_numpy_array = img_to_array(img)

                    if img_numpy_array.shape == img_dimensions:
                        img_numpy_array = img_numpy_array.flatten()
                        img_array = np.append(img_array,img_numpy_array)
                        imgs_processed = imgs_processed + 1

                        ## loading info
                        percent_conv = (imgs_processed / timeout_val) * 100
                        print(str(percent_conv) + " percent converted to pixel array")
                elif imgs_processed == timeout_val:
                    break

    # reshape into legal dimensions
    img_array = np.reshape(img_array,(imgs_processed,int(img_array.size/imgs_processed)))

    adapted_dataset = adapted_dataset.loc[matching_ids]

    # determine activation function (relu or tanh) from if there are negative numbers in target variable
    df_values = adapted_dataset.values
    df_values = df_values.flatten()
    for val in df_values:
        if val < 0:
            negative_vals = True
        else: negative_vals = False

    if negative_vals == True:
        act_func = "tanh"
    else:
        act_func = 'relu'

    def model(pd_data,input_imagery,target_var,activation_function):

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

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        X_train = X_train.to_numpy()
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Image

        X_train_img, X_test_img = train_test_split(input_imagery,test_size=0.2,random_state=42)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        print(X_train_img.shape)
        print(X_train.shape)
        data = np.concatenate((X_train_img,X_train),axis=1)
        data_test = np.concatenate((X_test,X_test_img),axis=1)

        scaler = StandardScaler().fit(data)
        data = scaler.transform(data)
        data_test = scaler.transform(data_test)

        print(activation_function)

        # set input shape to dimension of data
        input = keras.layers.Input(shape=(data.shape[1],))

        x = Dense(1100, activation=activation_function)(input)
        x = Dense(1000, activation=activation_function)(x)
        x = Dense(1000, activation=activation_function)(x)
        x = Dense(750, activation=activation_function)(x)
        x = Dense(750, activation=activation_function)(x)
        x = Dense(500, activation=activation_function)(x)
        x = Dense(500, activation=activation_function)(x)
        x = Dense(100, activation=activation_function)(x)
        x = Dense(100, activation=activation_function)(x)
        x = Dense(50, activation=activation_function)(x)
        x = Dense(50, activation=activation_function)(x)
        x = Dense(25, activation=activation_function)(x)
        x = Dense(10, activation=activation_function)(x)
        output = Dense(1, activation='linear')(x)
        model = keras.Model(input, output)

        model.compile(optimizer=keras.optimizers.SGD(momentum=0.8,learning_rate=0.01),
                          loss='mean_squared_error',
                          metrics=['accuracy'])

        fit = model.fit(data,y_train,epochs=10,batch_size=5)

        history = fit

        def plot(model_history, metric, graph_title):
            history = model_history
            plt.plot(history.history[metric])
            plt.title(graph_title)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.show()

        plot(history, 'accuracy', 'model accuracy')
        plot(history, 'loss', 'model loss')

        print(model.predict(data_test))
        print(y_test)

        def save_fitted_model(model, save_location):
            model.save(save_location)

        if save_fit == True:
            save_fitted_model(model, model_save_loc)

    model(adapted_dataset,img_array,target_var,act_func)

if run_img_model == True:
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