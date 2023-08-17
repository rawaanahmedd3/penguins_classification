import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

def preprocessing(filename):
    missing_values = ["na", "N/A", np.nan]
    df = pd.read_csv(filename, na_values=missing_values, low_memory=False)

    for col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    label_encoder = LabelEncoder()
    df['gender'] = label_encoder.fit_transform(df['gender'])

    return df

#def inputs(df, f1, f2, c1, c2, b):
def inputs(df):
    #features_dic = {'bill_length': 'bill_length_mm', 'bill_depth': 'bill_depth_mm', 'flipper_length': 'flipper_length_mm', 'gender': 'gender', 'body_mass': 'body_mass_g'}
    features_dic = {'1': 'bill_length_mm', '2': 'bill_depth_mm', '3': 'flipper_length_mm', '4': 'gender', '5': 'body_mass_g'}
    f1, f2 = input("choose only 2 features out of 5 (1.bill_length_mm,2.bill_depth_mm,3.flipper_length_mm,4.gender,5.body_mass_g):").split()
    if f1 in features_dic and f2 in features_dic:
        dataframe = pd.DataFrame(columns=['species', features_dic[f1], features_dic[f2]])

    c1, c2 = input("choose only 2 classes out of 3 (1.adelie,2.gentoo,3.chinstrap):").split()
    #if c1 == "Adelie" and c2 == "Gentoo":
    if c1 == "1" and c2 == "2":
        rows = df.iloc[:100, [0, int(f1), int(f2)]]
        dataframe = dataframe.append(rows, ignore_index=True)
    #if c1 == "Adelie" and c2 == "Chinstrap":
    if c1 == "1" and c2 == "3":
        rows1 = df.iloc[:50, [0, int(f1), int(f2)]]
        rows2 = df.iloc[100:150, [0, int(f1), int(f2)]]
        rows3 = [rows1, rows2]
        rows = pd.concat(rows3)
        dataframe = dataframe.append(rows, ignore_index=True)
    #if c1 == "Gentoo" and c2 == "Chinstrap":
    if c1 == "2" and c2 == "3":
        rows = df.iloc[50:150, [0, int(f1), int(f2)]]
        dataframe = dataframe.append(rows, ignore_index=True)

    b = input("do you want to add bias or not?:")
    if b == "yes":
        value = random.random()
    else:
        value = 0

    learning_rate = float(input("enter learning rate (0<LR<=1):"))

    epochs_num = input("enter number of epochs:")

    weight1 = random.random()
    weight2 = random.random()

    return dataframe, value, learning_rate, epochs_num, weight1, weight2


def train_test_data(df):
    label_encoder = LabelEncoder()
    df['species'] = label_encoder.fit_transform(df['species'])

    # split data
    class1 = df.iloc[:50, :]
    labels1 = class1['species']
    features1 = class1.iloc[:, 1:3]
    l1_train, l1_test, f1_train, f1_test = train_test_split(labels1, features1, train_size=0.6, random_state=0)

    class2 = df.iloc[50:, :]
    labels2 = class2['species']
    features2 = class2.iloc[:, 1:3]
    l2_train, l2_test, f2_train, f2_test = train_test_split(labels2, features2, train_size=0.6, random_state=0)

    # training dataframe
    train_data1 = pd.Series
    train_data1 = pd.concat([l1_train, l2_train])
    train_data2 = pd.concat([f1_train, f2_train])
    train_data = pd.concat([train_data1, train_data2], axis=1)

    # testing dataframe
    test_data1 = pd.Series
    test_data1 = pd.concat([l1_test, l2_test])
    test_data2 = pd.concat([f1_test, f2_test])
    test_data = pd.concat([test_data1, test_data2], axis=1)

    return train_data, test_data


def signum(num):  # activation function
    if num > 0:
        return 1
    else:
        return 0

def perceptron(tdf, w1, w2, eta, epoch, bias):  # single layer perceptron function
    s = 0
    for e in epoch:
        for row, column in tdf.iterrows():
            t = column[0]
            f1 = column[1]
            f2 = column[2]
            sum = (f1 * w1) + (f2 * w2) + bias
            pred = signum(sum)
            if pred != t:
                error = t - pred
                w1 = w1 + (eta * error * f1)
                w2 = w2 + (eta * error * f2)
            else:
                s += 1
        if s / 60 == 1:
            break
        else:
            continue
    return w1, w2

def test(tdf, u_w1, u_w2, bias):
    target_list = []
    predicted_list = []
    for row, column in tdf.iterrows():
        t = column[0]
        f1 = column[1]
        f2 = column[2]
        sum = (f1 * u_w1) + (f2 * u_w2) + bias
        pred = signum(sum)
        target_list.append(t)
        predicted_list.append(pred)
    # build confusion matrix
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for a, p in zip(target_list, predicted_list):
        if p == a:  # correct prediction (t)
            if p == 1:  # correct for first class (tp)
                tp += 1
            else:  # correct for second class (tn)
                tn += 1
        else:  # incorrect prediction (f)
            if p == 1:  # incorrect for first class (fp)
                fp += 1
            else:  # incorrect for second class (fn)
                fn += 1
    matrix = np.array([[tp, fn], [tn, fp]])
    acc = (tp + tn) / (len(target_list))

    return matrix, acc

def line_equation(dataframe, u_w1, u_w2, bias):
    # line equation w1x1+w2x2+b=0 , x1=0 > x2=-b/w2 , x2=0 > x1=-b/w1
    # then two points of boundary line are:
    point1 = [0, -bias / u_w2]  # x=0, y=value
    point2 = [-bias / u_w1, 0]  # x=value, y=0
    plt.plot(point1, point2)
    plt.show()

    # visualize of training data
    plt.scatter(dataframe.iloc[0:30, [1]], dataframe.iloc[0:30, [2]], color='orange')
    plt.scatter(dataframe.iloc[30:60, [1]], dataframe.iloc[30:60, [2]], color='green')
    plt.show()

    #both together
    col = dataframe.iloc[:, 1]
    min_x = min(col)
    max_x = max(col)
    min_y = (-(u_w1*min_x)-bias)/u_w2
    max_y = (-(u_w1*max_x)-bias)/u_w2

    plt.plot((min_x, max_x), (min_y, max_y))
    plt.scatter(dataframe.iloc[0:30, [1]], dataframe.iloc[0:30, [2]], color='orange')
    plt.scatter(dataframe.iloc[30:60, [1]], dataframe.iloc[30:60, [2]], color='green')
    plt.show()

#def main(f1, f2, c1, c2, lr, e, b):
def main():
    df = preprocessing("penguins.csv")
    #new_df, bias, w1, w2 = inputs(df, f1, f2, c1, c2, b)
    new_df, bias, lr, e, w1, w2 = inputs(df)
    training_data, testing_data = train_test_data(new_df)
    new_w1, new_w2 = perceptron(training_data, w1, w2, lr, e, bias)
    confusion_matrix, accuracy = test(testing_data, new_w1, new_w2, bias)
    print(confusion_matrix)
    print(accuracy)
    line_equation(training_data, new_w1, new_w2, bias)

main()
