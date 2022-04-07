import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# For data preparation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve

# For model evaluation
from sklearn import metrics

# For ML classification
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# For Tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Conv1D, Dropout
from tensorflow.keras import Model
from tensorflow.math import confusion_matrix

# for callback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# ============================== extract ==============================================================

def extract_capteur(data, samples) :
    data_capteur1 = []
    data_capteur2 = []
    data_capteur3 = []
    data_capteur4 = []
    data_capteur5 = []
    data_capteur6 = []
    data_capteur7 = []
    data_capteur8 = []
    for i in range(8) :
        for j in range(0,samples) :
            if i==0 : 
                data_capteur1.append(list(data[8*j]))
            elif i==1 :
                data_capteur2.append(list(data[i + 8*j]))
            elif i==2 :
                data_capteur3.append(list(data[i + 8*j]))
            elif i==3 :
                data_capteur4.append(list(data[i + 8*j]))
            elif i==4 :
                data_capteur5.append(list(data[i + 8*j]))
            elif i==5 :
                data_capteur6.append(list(data[i + 8*j]))
            elif i==6 :
                data_capteur7.append(list(data[i + 8*j]))
            elif i==7 :
                data_capteur8.append(list(data[i + 8*j]))
    return pd.DataFrame(data_capteur1),pd.DataFrame(data_capteur2),pd.DataFrame(data_capteur3),pd.DataFrame(data_capteur4),pd.DataFrame(data_capteur5),pd.DataFrame(data_capteur6),pd.DataFrame(data_capteur7),pd.DataFrame(data_capteur8)

# ============================== Labelling ==============================================================

def label(data_capteur1,data_capteur2,data_capteur3,data_capteur4,data_capteur5,data_capteur6,data_capteur7,data_capteur8, label) :
    data_capteur1.insert(4000,"class",label)
    data_capteur2.insert(4000,"class",label)
    data_capteur3.insert(4000,"class",label)
    data_capteur4.insert(4000,"class",label)
    data_capteur5.insert(4000,"class",label)
    data_capteur6.insert(4000,"class",label)
    data_capteur7.insert(4000,"class",label)
    data_capteur8.insert(4000,"class",label)


# ============================== KNN ==============================================================


def knncapteur(data, test_size) : 
    features = data.copy()
    etiq = features["class"]
    features.drop(["class"], axis=1, inplace=True)
    fv_train, fv_test, etiq_train, etiq_test = train_test_split(features, etiq, test_size=test_size,random_state=42)
    parameters = {'n_neighbors':np.arange(1,20,1), 'p' : np.arange(1,3,1)}
    knn=KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters)
    clf.fit(fv_train,etiq_train)
    knn=KNeighborsClassifier(**clf.best_params_)
    knn.fit(fv_train,etiq_train)

    N, train_score, val_score = learning_curve(knn, fv_train, etiq_train, train_sizes=np.linspace(0.1, 1.0, 10), cv = 5)

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10,8))
    plt.plot(N, train_score.mean(axis=1), label='Train')
    plt.plot(N, val_score.mean(axis=1), label='Validation')
    plt.xlabel('Train sizes')
    plt.title('Learning curve')
    plt.legend()

    yknn=knn.predict(fv_test)
    print("Accuracy : ",metrics.accuracy_score(etiq_test, yknn))
    print("F1-score KNN : ", metrics.f1_score(etiq_test, yknn, average='macro'))
    print(metrics.classification_report(etiq_test, yknn))

    plt.figure(figsize=(8,6))
    sns.heatmap(metrics.confusion_matrix(etiq_test, yknn), annot=True, annot_kws={"size": 10})
    plt.title('Confusion matrix KNN')



# ============================== SVM ==============================================================

def svmcapteur(data, test_size) :
    features = data.copy()
    etiq = features["class"]
    features.drop(["class"], axis=1, inplace=True)
    fv_train, fv_test, etiq_train, etiq_test = train_test_split(features, etiq, test_size=test_size,random_state=42)

    parameters = {'degree':np.arange(1,20,1)}
    svm=SVC(gamma='scale', class_weight='balanced')
    clf = GridSearchCV(svm, parameters)
    clf.fit(fv_train,etiq_train)
    svm=SVC(**clf.best_params_)
    svm.fit(fv_train,etiq_train)

    N, train_score, val_score = learning_curve(svm, fv_train, etiq_train, train_sizes=np.linspace(0.1, 1.0, 10), cv = 5)
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10,8))
    plt.plot(N, train_score.mean(axis=1), label='Train')
    plt.plot(N, val_score.mean(axis=1), label='Validation')
    plt.xlabel('Train sizes')
    plt.title('Learning curve')
    plt.legend()

    ysvm=svm.predict(fv_test)
    print("Accuracy : ",metrics.accuracy_score(etiq_test, ysvm))
    print("F1-score SVM : ",metrics.f1_score(etiq_test, ysvm, average='macro'))
    print(metrics.classification_report(etiq_test, ysvm))
    
    plt.figure(figsize=(8,6))
    sns.heatmap(metrics.confusion_matrix(etiq_test, ysvm), annot=True, annot_kws={"size": 10})
    plt.title('Confusion matrix SVM')


# ============================== CNN ==============================================================
# TODO : A SUBDIVISER EN PLUSIEURS SOUS FONCTIONS et ajouter plus de modularité pour choisir directement de l'extérieur de la fonction


def cnn1D(data, test_size=0.2) :
    # Préparation des données
    features = data.copy()
    etiq = features['class']
    features.drop(['class'], inplace=True, axis=1)
    etiq = etiq.astype('category').cat.codes
    features_train, features_test, labels_train, labels_test = train_test_split(features, etiq, test_size=test_size,random_state=42)

    # Normalisation
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(np.transpose(features_train))
    features_train = np.transpose(scaler.transform(np.transpose(features_train)))
    scaler2 = MinMaxScaler(feature_range=(0,1))
    scaler2.fit(np.transpose(features_test))
    features_test = np.transpose(scaler2.transform(np.transpose(features_test)))

    # Reshaping 
    features_train = np.expand_dims(features_train, axis=2)
    features_test = np.expand_dims(features_test, axis=2)

    labels_train = np.expand_dims(labels_train, axis=1)
    labels_test = np.expand_dims(labels_test, axis=1)
    labels_train.shape, labels_test.shape

    # CNN Variables
    # Première couche du CNN
    filter_size1 = 5
    num_filters1 = 64
    # Deuxième couche du CNN 
    filter_size2 = 5
    num_filters2 = 32

    batch_size = 4
    num_channels = 1 # ! paramètre sur lequel on pourrait jouer si on ajoute les autres capteurs
    n_epochs = 50
    

    # CNN
    path_to_save_model = './Model_CNN1D'
    ckpt_saver = ModelCheckpoint(
    path_to_save_model,
    monitor='accuracy', # sur quoi on se base pour voir le meilleur
    mode = 'max', # max de l'accuracy sur la validation
    save_best_only = True,
    save_freq='epoch', # ne voit qu'à la fin de l'époque
    verbose=1
    ) 


    model = tf.keras.Sequential(
    [
        Input(shape=(4000,num_channels)), # format d'entrée
        Conv1D(filters=num_filters1, kernel_size=filter_size1, activation='relu'),
        Conv1D(filters=num_filters2, kernel_size=filter_size2, activation='relu'),
        Dropout(0.5),
        MaxPool1D(pool_size=2),
        Conv1D(filters=num_filters1, kernel_size=filter_size1, activation='relu'),
        Conv1D(filters=num_filters2, kernel_size=filter_size1, activation='relu'),
        Dropout(0.5),
        MaxPool1D(pool_size=2),
        GlobalAvgPool1D(),
        Dense(50, activation='relu'),
        Dense(4, activation='softmax')
    ]
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
    model.fit(features_train, labels_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.2, callbacks=[ckpt_saver])

    # Evaluation of the model
    print("====== Modele evaluation ======")
    model.evaluate(features_test,labels_test, batch_size=batch_size)
    print("===============================")
    y_model=model.predict(features_test)
    y_model_max = np.argmax(y_model, axis=1)
    con_mat = tf.math.confusion_matrix(labels=labels_test, predictions=y_model_max).numpy()

    # Visualization
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(8, 6))
    sns.heatmap(con_mat, annot=True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return y_model, y_model_max