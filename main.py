import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
import matplotlib.pyplot as plt
import pydicom as dicom
import os
import cv2
from PIL import Image, ImageFilter

save_fit = False
model_save_loc = "saved_model"

main_data = "Patient and Treatment Characteristics.csv"
sec_data = "HNSCC Clinical Data.csv"
test_file = "test_2.csv"
target_variable = "PostRT Skeletal Muscle status"

#if true, dicom images will be converted to png instead of jpg
png_images = False

#list of names for duplicate columns
var_blacklist = ["Gender","Age at Diag"]

load_dir = "D:\Cancer Imagery\HNSCC"

save_dir = "D:\converted_img"

num_epochs = 80

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
        print(y_test)

    NN(adapted_dataset,target_variable,epochs_num)

#model(main_data,test_file,target_variable,num_epochs)

# initialize lists for annotation data
patient_ids = []
study_dates = []

#input file location. Converts image to digits
def image_digitizer(dir):

    im = Image.open(dir).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))
        newImage.paste(img, (wleft, 4))

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

# collect directory for every image in data
def collect_image_dirs(patient_data_loc):
    img_directories = []

    Patients = os.listdir(patient_data_loc)

    for patients in Patients:
        patient_datasets = os.listdir(patient_data_loc + "\\" + patients)
        dataset = patient_datasets[0]
        data_dir = os.listdir(patient_data_loc + "\\" + patients + "\\" + dataset)
        data = data_dir[0]
        data_img = os.listdir(patient_data_loc + "\\" + patients + "\\" + dataset + "\\" + data)
        img = data_img[0]
        img_directories.append(patient_data_loc + "\\" + patients + "\\" + dataset + "\\" + data + "\\" + img)

    return img_directories

img_directories = collect_image_dirs(load_dir)
print(img_directories)
print(len(img_directories))

def image_prep(png_boolean,image_path,save_path):
    png = png_boolean

    ds = dicom.read_file(image_path)

    #append annotation data to lists
    patient_id = ds.PatientID
    patient_ids.append(patient_id)
    study_date = ds.StudyDate
    study_dates.append(study_date)

    pixel_array_numpy = ds.pixel_array
    if png == False:
        image = image_path.replace(".dcm", ".jpg")
    elif png == True:
        image = image_path.replace(".dcm", ".png")
    cv2.imwrite(image, pixel_array_numpy)

for img in img_directories:
    image_prep(png_images,img,save_dir)

# Find names of every image suitable for use (png or jpg format)
def collect_images(images_loc,images_format):
    usable_images_names = []
    files_in_dir = os.listdir(images_loc)
    for i in files_in_dir:
        file_format = i[-4:]
        if file_format == images_format:
            usable_images_names.append(i)
    return usable_images_names

# define "images_format" parameter for function use
if png_images == False:
    image_format = ".jpg"
elif png_images == True:
    image_format = ".png"

images_list = collect_images(save_dir,image_format)

data = pd.DataFrame()

def ID_append(digitized_image,ID):
    digitized_image.insert(0,ID)
    return digitized_image

i = 0
for images in images_list:
    dig_img = image_digitizer("converted_img\\"+images)
    dig_img = ID_append(dig_img,int((patient_ids[i])[-4:]))
    data.insert(0,i,dig_img)
    i = i + 1

data = data.transpose()
data = data.rename(columns={data.columns[0]:"ID"})
print(data)