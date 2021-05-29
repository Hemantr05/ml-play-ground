import streamlit as st
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def add_preprocess_parameter_ui(preprocessor):
    """
        Preprocesses the dataset
    """
    params = dict()
    if preprocessor == 'Standard Scaler':
        ## TODO: Implement Standard Scaler
        pass
    else:
        pass

    return params


def add_model_parameter_ui(clf_name):
    """
        Returns model parameters based on widget
    """
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider("K", 1, 5)
        params["K"] = K

    elif clf_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators

    else:
        pass

    return params


def get_classifier(clf_name, params):
    """
        Initializes selected model with its corresponding parameters
    """
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == 'SVM':
        clf = SVC(C=params["C"])

    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth = params["max_depth"], random_state=1234)

    else:
        clf = LogisticRegression()

    return clf

def get_dataset(dataset_name):
    """
        Loads and returns selected dataset
    """
    if dataset_name == "Iris":
        data = datasets.load_iris()

    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()

    elif dataset_name == "MNIST":
        data = datasets.load_digits()

    #elif dataset_name == "Boston Housing Price":
    #    data = datasets.load_boston()

    X = data.data
    y = data.target

    return X, y


def get_metrics(metric, y_pred, y_test):
    """
        Returns selected metric value
    """
    if metric == "Accuracy":
        result = accuracy_score(y_test, y_pred)
    else:
        ## Throwing an error
        result = roc_auc_score(y_true=y_test, y_score=y_pred, multi_class='ovo')

    return result

def visualize(vis, features, label):
    """
        Helper funtion for visualization
    """
    if vis == 'PCA':
        #n_components = st.sidebar.slider("n_components", 2, 10)
        #alpha = st.sidebar.slider("alpha", 0.8, 2.0)
        #pca = PCA(n_components)
        pca = PCA(2)

        X_projected = pca.fit_transform(features)
        
        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]


        fig = plt.figure()
        plt.scatter(x1, x2, c=label, alpha=0.8, cmap='viridis')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()

        st.pyplot()