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
import shutil
import cv2
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import psutil
import sys
import tkinter as tk
import tkinter.font as tkFont
import random
from tkinter import ttk
import GUI
from statistics import mean

# un-comment to show all of pandas dataframe
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

# un-comment to show all of numpy array
#np.set_printoptions(threshold=sys.maxsize)

# if true, main GUI will be used to specify other varibles
useFront = False

if useFront == False:
    # SPECIFY VARIABLES HERE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    save_fit = False
    model_save_loc = "saved_model"

    main_data = "D:\\Cancer_Project\\Team8_Cancer_ML\HNSCC-HN1\\Copy of HEAD-NECK-RADIOMICS-HN1 Clinical data updated July 2020 (original).csv"
    sec_data = ""
    test_file = "test_2.csv"

    # list with strings or a single string may be inputted
    target_variables = 'chemotherapy_given'

    # if true, converted images will be in png format instead of jpg
    png = False

    # folder containing Cancer Imagery
    load_dir = "D:\Cancer_Project\\Cancer Imagery\\HNSCC"

    # directory to save data such as converted images
    save_dir = "D:\\Cancer_Project\\converted_img"

    # directory to save imagery array
    img_array_save = "D:\Cancer_Project\img_arrays"

    # if true, numpy image array will be searched for in img_array_save
    load_numpy_img = False

    # if true, attempt will be made to convert dicom files to jpg or png
    convert_imgs = False

    #if true, converted dicom images will be deleted after use
    del_converted_imgs = False

    # if true, image model will be ran instead of clinical only model
    run_img_model = False

    # if true, two data files will be expected for input
    two_datasets = False

    # if true, an additional file will be expected for testing
    use_additional_test_file = False

    # where image id is located in image names (start,end)
    # only applies if using image model
    img_id_name_loc = (3,6)

    # Column of IDs in dataset. Acceptable values include "index" or a column name.
    ID_dataset_col = "id"

    # tuple with dimension of imagery. All images must equal this dimension
    img_dimensions = (512, 512, 3)

    # if true, every column in data will be inputted for target variable
    target_all = False

    # save location for data/graphs
    data_save_loc = "D://Cancer_Project//Team8_Cancer_ML//result_graphs"

    # if true, graphs will be shown after training model
    show_figs = False

    # if true, graphs will be saved after training model
    save_figs = False

    # number of epochs in model
    num_epochs = 50

    # END VARIABLES - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
elif useFront == True:

    boolList = GUI.boolList

    # convert every element in boolList to a proper boolean
    [bool(b) for b in boolList]

    dictTxt = dict(zip(GUI.varList_txt,GUI.txtEntry_list))
    dictBool = dict(zip(GUI.varList_bool,boolList))

    save_fit = dictBool["save_fit "]
    model_save_loc = dictTxt["model_save_loc "]

    main_data = dictTxt["main_data "]
    sec_data = dictTxt["sec_data "]
    test_file = dictTxt["test_file "]

    # list with strings or a single string may be inputted
    # check if string is list. Find returns -1 if value cannot be found
    if dictTxt["target_variables "].find("[") != -1 and dictTxt["target_variables "].find(",") != -1:
        target_variables = list(dictTxt["target_variables "][1:-1].split(","))

        # remove excess quotes
        target_variables = ([v.strip("'") for v in target_variables])
        target_variables = ([v.replace("'",'') for v in target_variables])
    else:
        target_variables = dictTxt["target_variables "]

    # if true, converted images will be in png format instead of jpg
    png = dictBool["png "]

    # folder containing Cancer Imagery
    load_dir = dictTxt["load_dir "]

    # directory to save data such as converted images
    save_dir = dictTxt["save_dir "]

    # directory to save imagery array
    img_array_save = dictTxt["img_array_save "]

    # if true, numpy image array will be searched for in img_array_save
    load_numpy_img = dictBool["load_numpy_img "]

    # if true, attempt will be made to convert dicom files to jpg or png
    convert_imgs = dictBool["convert_imgs "]

    #if true, converted dicom images will be deleted after use
    del_converted_imgs = dictBool["del_converted_imgs "]

    # if true, image model will be ran instead of clinical only model
    run_img_model = dictBool["run_img_model "]

    # if true, two data files will be expected for input
    two_datasets = dictBool["two_datasets "]

    # if true, an additional file will be expected for testing
    use_additional_test_file = dictBool["use_additional_test_file "]

    # where image id is located in image names (start,end)
    # only applies if using image model
    img_id_name_loc = dictTxt["img_id_name_loc "]

    # Column of IDs in dataset. Acceptable values include "index" or a column name.
    ID_dataset_col = dictTxt["ID_dataset_col "]

    # tuple with dimension of imagery. All images must equal this dimension
    img_dimensions = dictTxt["img_dimensions "]

    # if true, every column in data will be inputted for target variable
    target_all = dictBool["target_all "]

    # save location for data/graphs
    data_save_loc = dictTxt["data_save_loc "]

    # if true, graphs will be shown after training model
    show_figs = dictBool["show_figs "]

    # if true, graphs will be saved after training model
    save_figs = dictBool["save_figs "]

    # number of epochs in model
    num_epochs = int(dictTxt["num_epochs "])

def encodeText(dataset):

    if str(type(dataset)) == "<class 'str'>":
        dataset = pd.read_csv(dataset)

    dShape = dataset.shape
    axis1 = dShape[0]
    axis2 = dShape[1]

    if axis1 >= axis2:
        longestAxis = axis1
        shortestAxis = axis2
    else:
        longestAxis = axis2
        shortestAxis = axis1

    for i in range(longestAxis):
        for n in range(shortestAxis):
            if longestAxis == axis1:
                data = dataset.iloc[i,n]
            else:
                data = dataset.iloc[n,i]

            if str(type(data)) == "<class 'str'>":
                strData = ""
                for c in data:
                    cInt = ord(c)
                    strData = strData + str(cInt)

                strData = int(strData)

                if longestAxis == axis1:
                    dataset.iloc[i,n] = strData
                else:
                    dataset.iloc[n,i] = strData

    return dataset

encodedDataset = encodeText(main_data)

def GUI_varConnector(dataset1, dataset2):

    if str(type(dataset1)) == "<class 'str'>":
        dataset1 = pd.read_csv(dataset1)

    if str(type(dataset2)) == "<class 'str'>":
        dataset2 = pd.read_csv(dataset2)

    vars1 = list(dataset1.columns)
    vars2 = list(dataset2.columns)

    vars1.remove(ID_dataset_col)
    vars2.remove(ID_dataset_col)

    for element in target_variables:
        if element in vars1:
            vars1.remove(element)
        if element in vars2:
            vars2.remove(element)

    # list of colors for buttons to choose from
    colors = ["red", "blue", "purple", "orange", "green", "gray",
              "gainsboro", "dark salmon", "LemonChiffon2", "ivory3",
              "SteelBlue1", "DarkOliveGreen3", "gold2", "plum1"]

    window = tk.Tk()

    main_frame = tk.Frame(window)
    main_frame.pack(fill=tk.BOTH,expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a scrollbars to the canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    scrollbar_x = ttk.Scrollbar(main_frame,orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Configure the canvas
    canvas.configure(xscrollcommand=scrollbar_x.set)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    second_frame = tk.Frame(canvas)
    canvas.create_window((0,0), window=second_frame, anchor="nw")

    buttonFont = tkFont.Font(family="Georgia", size=20)
    font = tkFont.Font(family="Georgia",size=25)
    title = tk.Label(text="Select matching variables", font=font, fg="#0352fc")
    title.place(relx=0.2,rely=0)

    button = None

    pressedVars = []
    buttonList = []

    def makeButtons(var_name, x, y):
        var = var_name

        def trackVars():
            pressedVars.append(var)
            button.config(bg=random.choice(colors))

        button = tk.Button(master=second_frame,text=var_name, fg="white", bg="black", width=30, height=1,
                           command=trackVars,font=buttonFont)
        button.grid(column=x,row=y,padx=105,pady=50)
        buttonList.append(button)

    y = 1
    for var in vars1:
        makeButtons(var, 10, y)
        y = y + 10

    y = 1
    for var2 in vars2:
        makeButtons(var2, 20, y)
        y = y + 10

    exitButton = tk.Button(master=second_frame,text="Done",fg="white",bg="orange",width=30,height=3,
                           command=window.destroy)
    exitButton.grid(row=1,column=100)

    window.mainloop()

    # function used to convert list to dictionary
    def Convert(lst):
        res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
        return res_dct

    pressedVars_dict = Convert(pressedVars)
    return pressedVars_dict

if two_datasets == True:
    varMatches = GUI_varConnector(main_data,sec_data)
    print(varMatches)

def collect_img_dirs(data_folder):
    img_directories = []

    for root, dirs, files, in os.walk(data_folder):
        for name in files:
            dir = os.path.join(root,name)
            img_directories.append(dir)

    return img_directories

if convert_imgs == True:
    load_dirs = collect_img_dirs(load_dir)

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

        # eliminate duplicate variables
        for i in varMatches.values():
            combined_dataset = combined_dataset.drop(i,axis=1)
        data = combined_dataset
    else:
        data = file_1

    return data

if two_datasets == True:
    main_data = prep_data(main_data,sec_data)
elif two_datasets == False:
    main_data = prep_data(main_data,None)

resultList = []
prediction = []

def model(data_file, test_file, target_vars, epochs_num):

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
            dataset = dataset.assign(target_variables=y_data)
            target_name = str(target_var)
            dataset = dataset.rename(columns={'target_variables':target_name},inplace=True)
        elif use_additional_test_file == False:
            dataset = df

        return dataset

    adapted_dataset = format_data(data_file, test_file, target_vars)

    # initiate negative_vals as False
    negative_vals = False

    # determine activation function (relu or tanh) from if there are negative numbers in target variable
    df_values = adapted_dataset.values
    df_values = df_values.flatten()
    for val in df_values:
        val = float(val)
        if val < 0:
            negative_vals = True

    if negative_vals == True:
        act_func = "tanh"
    else:
        act_func = 'relu'

    print(act_func)

    def NN(data_file, target_vars, epochs_num,activation_function):
        global resultList
        global prediction

        # Get data. Data must already be in a Pandas Dataframe
        df = data_file

        # round all values in dataset to 3rd decimal place
        df = df.astype("float").round(3)

        #y data
        labels = df.loc[:,target_vars]
        #x data
        features = df.drop(target_vars,axis=1)

        X = features
        y = labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if str(type(target_vars)) == "<class 'list'>" and len(target_vars) > 1:
            # divide y_train's columns into separate dataframes and store them in a list
            list_y_train = []
            y_cols = list(y_train.columns)
            for col in y_cols:
                var_col = y_train[col]
                list_y_train.append(var_col)

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        # check data for nans/non-compatible objects
        def hasNan(array):
            nan = np.isnan(array)
            for arr in nan:
                if array.ndim == 2: 
                    for bool in arr:
                        if bool: 
                            containsNan = True
                        else: 
                            containsNan = False
                elif array.ndim == 1: 
                    if arr: 
                        containsNan = True
                    else: 
                        containsNan = False

            # check that all data is floats or integers 
            if array.ndim == 1: 
                typeList = []
                for vals in array: 
                    valType = str(type(vals))
                    typeList.append(valType)

                print(containsNan)
                for types in typeList: 
                    if types != "<class 'numpy.float64'>" and types != "<class 'numpy.int64'>": 
                        containsNan = True

            if containsNan: 
                print("Data contains nan values")
            else: 
                print("Data does not contain nan values")

        hasNan(y_train)

        if str(type(target_vars))=="<class 'list'>" and len(target_vars) > 1:
            input = keras.Input(shape=X_train.shape)

            def add_target(Input):
                x = layers.Dense(40,activation=activation_function)(Input)
                x = layers.Dense(40,activation=activation_function)(x)
                x = layers.Dense(35,activation=activation_function)(x)
                x = layers.Dense(35,activation=activation_function)(x)
                return x

            output_list = []
            for vars in range(len(target_vars)):
                x = add_target(input)
                output_list.append(x)

            x = layers.Concatenate()(output_list)
            output_list.clear()
            x = layers.Dense(12,activation='relu')(x)
            for vars in range(len(target_vars)):
                y = layers.Dense(1,activation='linear')(x)
                output_list.append(y)

            model = keras.Model(inputs=input,outputs=output_list)

            model.compile(optimizer='SGD',
                          loss='mean_absolute_error',
                          metrics=['accuracy'])

            print(list_y_train)

            fit = model.fit(X_train, list_y_train, epochs=epochs_num, batch_size=5)

        else:
            # set input shape to dimension of data
            input = keras.layers.Input(shape=(X_train.shape[1],))

            x = Dense(30, activation=activation_function)(input)
            x = Dense(30, activation=activation_function)(x)
            x = Dense(30, activation=activation_function)(x)
            x = Dense(25, activation=activation_function)(x)
            x = Dense(25, activation=activation_function)(x)
            x = Dense(25, activation=activation_function)(x)
            x = Dense(20, activation=activation_function)(x)
            x = Dense(20, activation=activation_function)(x)
            x = Dense(15, activation=activation_function)(x)
            x = Dense(10, activation=activation_function)(x)
            output = Dense(1, activation='linear')(x)
            model = keras.Model(input, output)

            model.compile(optimizer='SGD',
                          loss='mean_absolute_error',
                          metrics=['accuracy'])

            fit = model.fit(X_train, y_train, epochs=epochs_num, batch_size=512)

        # plotting
        history = fit

        def plot(model_history,metric,graph_title):
            history = model_history
            plt.plot(history.history[metric])
            plt.title(graph_title)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            if save_figs == True:
                plt.savefig(os.path.join(data_save_loc, str(target_vars) + " " + metric + ".jpg"))

            if show_figs == True:
                plt.show()
            else:
                plt.clf()

        plot(history,'loss','model accuracy')

        def save_fitted_model(model,save_location):
            model.save(save_location)

        if save_fit == True:
            save_fitted_model(model,model_save_loc)

        prediction = model.predict(X_test, batch_size=1)
        roundedPred = np.around(prediction,0)

        print("- - - - - - - - - - - - - Unrounded Prediction - - - - - - - - - - - - -")
        print(prediction)
        print("- - - - - - - - - - - - - Rounded Prediction - - - - - - - - - - - - -")
        print(roundedPred)
        print("- - - - - - - - - - - - - y test - - - - - - - - - - - - -")
        print(y_test)

        eval = model.evaluate(X_test)
        results = dict(zip(model.metrics_names, eval))
        print(results)

        resultList.append(str(prediction))
        resultList.append(str(roundedPred))
        resultList.append(str(y_test))

    NN(adapted_dataset, target_vars, epochs_num, act_func)

if run_img_model == False and target_all == False:
    model(encodedDataset,test_file,target_variables,num_epochs)
elif run_img_model == False and target_all == True:
    # collect columns in data
    cols = list(main_data.columns)
    for column in cols:
        model(main_data,test_file,column,num_epochs)

def image_model(save_loc,data_file,test_file,target_vars,epochs_num):
    print("starting image model")

    def format_data(data_file, test_file, target_vars):

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
            y_data = df[target_vars]
            dataset = dataset.assign(target_variables=y_data)
            target_name = str(target_vars)
            dataset.rename(columns={'target_variables':target_name},inplace=True)
        elif use_additional_test_file == False:
            dataset = df

        return dataset

    adapted_dataset = format_data(data_file, test_file,target_vars)
    adapted_dataset.index.names = ["ID"]

    img_array = np.array([])
    matching_ids = []
    img_list = os.listdir(save_loc)

    # number of images that match proper resolution
    num_usable_img = 0

    # used for loading info
    imgs_processed = 0

    if load_numpy_img == True:
        img_array = np.load(os.path.join(img_array_save,os.listdir(img_array_save)[0]))
        flat_res = int((img_dimensions[0]*img_dimensions[1]*img_dimensions[2])+1)
        num_img = int(img_array.shape[0]/flat_res)
        img_array = np.reshape(img_array,(num_img,flat_res))

        ## retrieving ids
        img_df = pd.DataFrame(data=img_array)
        cols = list(img_df.columns)
        id_col = img_df[cols[-1]].tolist()
        dataset_id = adapted_dataset.index.tolist()

        # determine what to put first in loop
        if len(id_col) >= len(dataset_id):
            longest = id_col
            shortest = dataset_id
        elif len(dataset_id) > len(id_col):
            longest = dataset_id
            shortest = id_col

        for id in longest:
            for id2 in shortest:
                if int(id) == int(id2):
                    matching_ids.append(id)

    elif load_numpy_img == False:

        for imgs in img_list:

            # find matching ids
            for ids in adapted_dataset.index:
                ids = int(ids)
                if ids == int(imgs[img_id_name_loc[0]:img_id_name_loc[1]]):
                    matching_ids.append(ids)
                    matching_ids = list(dict.fromkeys(matching_ids))

            # Collect/convert corresponding imagery
            print("starting data preparation process")
            for ids in matching_ids:
                if ids == int(imgs[img_id_name_loc[0]:img_id_name_loc[1]]):
                    img = load_img(os.path.join(save_loc, imgs))
                    img_numpy_array = img_to_array(img)
                    if img_numpy_array.shape == img_dimensions:
                        img_numpy_array = img_numpy_array.flatten()
                        img_numpy_array = np.insert(img_numpy_array,len(img_numpy_array),ids)
                        num_usable_img = num_usable_img + 1
                        img_array = np.append(img_array,img_numpy_array,axis=0)
                        imgs_processed = imgs_processed + 1

                    else:
                        matching_ids.remove(ids)

                ## Memory optimization
                if psutil.virtual_memory().percent >= 50:
                    break

                ## loading info
                total_img = len(img_list)
                percent_conv = (imgs_processed / total_img) * 100
                print(str(percent_conv) + " percent converted")
                print(str(psutil.virtual_memory()))

        # save the array
        np.save(os.path.join(img_array_save, "img_array"), img_array)

        # reshape into legal dimensions
        img_array = np.reshape(img_array,(num_usable_img,int(img_array.size/num_usable_img)))

    adapted_dataset = adapted_dataset.loc[matching_ids]

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

    def model(pd_data,input_imagery,target_vars,activation_function):
        global resultList
        global prediction

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Clinical
        # Get data
        df = pd_data

        # y data
        labels = df[target_vars]
        # x data
        features = df.drop(target_vars,axis=1)

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

        if len(target_vars) == 1:
            # set input shape to dimension of data
            input = keras.layers.Input(shape=(data.shape[1],))

            x = Dense(150, activation=activation_function)(input)
            x = Dense(150, activation=activation_function)(x)
            x = Dense(150, activation=activation_function)(x)
            x = Dense(120, activation=activation_function)(x)
            x = Dense(120, activation=activation_function)(x)
            x = Dense(100, activation=activation_function)(x)
            x = Dense(100, activation=activation_function)(x)
            x = Dense(80, activation=activation_function)(x)
            x = Dense(80, activation=activation_function)(x)
            x = Dense(45, activation=activation_function)(x)
            output = Dense(1, activation='linear')(x)
            model = keras.Model(input, output)

            model.compile(optimizer='adam',
                              loss='mean_squared_error',
                              metrics=['accuracy'])

            fit = model.fit(data,y_train,epochs=epochs_num,batch_size=5)

        else:
            input = keras.layers.Input(shape=(data.shape[1],))

            def add_target(Input):
                x = layers.Dense(40,activation=activation_function)(Input)
                x = layers.Dense(40, activation=activation_function)(x)
                x = layers.Dense(35, activation=activation_function)(x)
                x = layers.Dense(35, activation=activation_function)(x)
                return x

            output_list = []
            for vars in range(len(target_vars)):
                x = add_target(input)
                output_list.append(x)

            x = layers.Concatenate()(output_list)
            output_list.clear()
            x = layers.Dense(12,activation=activation_function)(x)
            for vars in range(len(target_vars)):
                # create output layer
                y = layers.Dense(1,activation='linear')(x)
                output_list.append(y)

            model = keras.Model(inputs=input,outputs=output_list)

            model.compile(optimizer='adam',
                          loss='mean_squared_error',
                          metrics=['accuracy'])

            fit = model.fit(data,y_train,epochs=epochs_num,batch_size=5)

        #plotting
        history = fit

        def plot(model_history, metric, graph_title):
            history = model_history
            plt.plot(history.history[metric])
            plt.title(graph_title)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            if save_figs == True:
                plt.savefig(os.path.join(data_save_loc, str(target_vars) + " " + metric + ".jpg"))

            if show_figs == True:
                plt.show()
            else:
                plt.clf()

        plot(history, 'loss', 'model loss')

        def save_fitted_model(model, save_location):
            model.save(save_location)

        if save_fit == True:
            save_fitted_model(model, model_save_loc)

        prediction = model.predict(X_test, batch_size=1)
        roundedPred = np.around(prediction,0)

        print("- - - - - - - - - - - - - Unrounded Prediction - - - - - - - - - - - - -")
        print(prediction)
        print("- - - - - - - - - - - - - Rounded Prediction - - - - - - - - - - - - -")
        print(roundedPred)
        print("- - - - - - - - - - - - - y test - - - - - - - - - - - - -")
        print(y_test)

        eval = model.evaluate(X_test)
        results = dict(zip(model.metrics_names, eval))
        print(results)

        resultList.append(str(prediction))
        resultList.append(str(roundedPred))
        resultList.append(str(y_test))

    model(adapted_dataset,img_array,target_vars,act_func)

if run_img_model == True and target_all == False:
    image_model(save_dir,main_data,test_file,target_variables,num_epochs)
elif run_img_model == True and target_all == True:
    # collect columns in data
    cols = list(main_data.columns)
    for column in cols:
        image_model(save_dir,main_data,test_file,target_variables,num_epochs)

def resultPage():
    root = tk.Tk()

    root.title("Results")
    root.iconbitmap("D:\Cancer_Project\Team8_Cancer_ML\cancer_icon.ico")

    # MAKE SCROLLBAR
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Add a scrollbars to the canvas
    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    # Configure the canvas
    canvas.configure(xscrollcommand=scrollbar_x.set)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    second_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=second_frame, anchor="nw")

    # define fonts
    titleFont = tkFont.Font(family="Georgia",size=20)
    titleColor = "#f29c2c"

    resultFont = tkFont.Font(family="Consolas",size=16)

    # ADD WIDGETS
    prediction = resultList[0]
    roundedPred = resultList[1]
    y_test = resultList[2]

    def placeResults(txt):
        result = tk.Label(second_frame,text=txt,font=resultFont,bg='black',fg='white')
        result.grid(pady=40)

    def destroy():
        root.quit()

    resultTitle = tk.Label(second_frame,text="Prediction",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(prediction)

    resultTitle = tk.Label(second_frame,text="Rounded Prediction",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(roundedPred)

    resultTitle = tk.Label(second_frame,text="y_test",font=titleFont,fg=titleColor)
    resultTitle.grid()

    placeResults(y_test)

    exitButton = tk.Button(second_frame,text="Exit",font=titleFont,fg=titleColor,command=destroy)
    exitButton.grid()

    root.mainloop()

resultPage()

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