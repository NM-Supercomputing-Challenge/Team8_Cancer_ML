import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
save_fit = False
model_save_loc = "saved_model"

main_data = "Patient and Treatment Characteristics.csv"
sec_data = "HNSCC Clinical Data.csv"
test_file = "test_2.csv"
target_variable = "PostRT Skeletal Muscle status"

#list of names for duplicate columns
var_blacklist = ["Gender","Age at Diag"]

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

model(main_data,test_file,target_variable,num_epochs)

















