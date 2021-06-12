import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from utils import (get_dataset,
                    get_classifier,
                    get_metrics,
                    add_preprocess_parameter_ui,
                    add_model_parameter_ui,
                    visualize)

st.title("Machine Learning Playground")


#To load custom dataset

data = st.sidebar.file_uploader("Upload")
extn = os.path.splitext(data)
if extn == '.csv':
    df = pd.read_csv(data)
elif extn == '.xlsx':
    df = pd.read_excel(data)

#st.write("""### %s Metadata """ %(data))
#st.write("Shape of dataset: ", df.shape)
#st.dataframe(df)



dataset = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "MNIST", "Boston Housing Price"))
## TODO: Fix Boston Housing Price Error

st.write(""" ### %s Metadata """ %(dataset))
X, y = get_dataset(dataset)
st.write("Shape of dataset: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))


preprocess = st.sidebar.selectbox("Preprocess", ("None", "Standard Scaler"))
preprocess_params = add_preprocess_parameter_ui(preprocess)

classifiers = st.sidebar.selectbox("Select Model", ("SVM", "Logistic Regression", "Random Forest", "KNN"))
clf_params = add_model_parameter_ui(classifiers)

clf = get_classifier(classifiers,clf_params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

metric = st.sidebar.selectbox("Select Metrics", ("Accuracy", "ROC_AUC"))
metrics = get_metrics(metric, y_pred, y_test)
st.write("Accuray: %f" %(metrics))


vis = st.sidebar.selectbox("Visualize", ("PCA",))
visualize(vis, X, y)
