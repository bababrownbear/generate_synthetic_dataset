# Heavily inspired by https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py

import pandas as pd
import numpy as np
import streamlit as st


def generate_synthetic_dataset(n_samples=10, 
                    n_classes=2, 
                    class_weights=[0.5, 0.5], 
                    n_features=1, 
                    feature_names=['EatsVeggies'], 
                    class_names=['Yes','No'], 
                    y_column_name="IsAmazing",
                    feature_weights=[0.7]):

    if n_samples is None:
        raise ValueError("n_samples cannot be None")
    if n_samples < 10:
        raise ValueError("n_samples should be greater than 10")
    if n_classes is None:
        raise ValueError("n_classes cannot be None")
    if n_classes < 2:
        raise ValueError("n_classes should be greater than or equal to 2")
    if n_features is None:
        raise ValueError("n_features cannot be None")
    if n_features < 1:
        raise ValueError("n_features should be greater than or equal to 1")
    if np.sum(class_weights) != 1.0:
        raise ValueError("class_weights must add up to 1.0")
    if feature_names is None:
        raise ValueError("feature_names cannot be None")
    if len(feature_names) != n_features:
        raise ValueError("n_features must equal length of feature_names")
    if feature_weights is None:
        raise ValueError("feature_weights cannot be None")
    if len(feature_weights) != n_features:
        raise ValueError("n_features must equal length of feature_weights")
    feature_weights_array = np.asarray(feature_weights)
    if (feature_weights_array > 1.0).sum() > 0:
        raise ValueError('feature_weight cannot be greater than 1.0')


    X = np.zeros(shape=(n_samples, n_features))
    X = pd.DataFrame(data=X, columns=feature_names)

    y = np.zeros((n_samples, 1))
    y = pd.DataFrame(data=y, columns=[y_column_name])

    start = 0
    stop = 0
    for i in range(n_classes):
        n_samples_for_class = int(n_samples * class_weights[i])
        stop = stop + n_samples_for_class
        y[start:stop] = class_names[i] 
        start = stop

    for ndx, col in enumerate(X.columns,0):
        col_weight = feature_weights[ndx]
        n_samples_for_class = int(n_samples * class_weights[0])
        num_positive_towards_class = int(n_samples_for_class * col_weight)
        X[X.columns[ndx]][0:num_positive_towards_class] = 'Yes'
        X[X.columns[ndx]][num_positive_towards_class:] = 'No'

    dataset = pd.DataFrame(data=pd.concat([X, y], axis=1))

    return dataset


if __name__ == '__main__':
    n_samples = st.sidebar.slider(str.format('Number of samples'), 10, 100, step=10)

    class_weight = st.sidebar.slider('Ratio of classes', 0.0, 1.0, step=0.1, value=0.8)

    st.sidebar.text_input('Enter column name for y', value='IsAmazing')

    n_classes = 2

    # class_names = []

    # class_name = st.sidebar.text_input("Enter class name for class #1",value="Yes")
    # class_names.append(class_name)

    # class_name = st.sidebar.text_input("Enter class name for class #1",value="No")
    # class_names.append(class_name)
    
    n_features = st.sidebar.slider('Number of features', 1, 10, step=1)

    feature_names = []
    feature_weights = []
    for feature_ndx in range(n_features):
        feature_name = st.sidebar.text_input("Enter feature name for feature #{feature_ndx}".format(feature_ndx=feature_ndx+1), value="EatsVeggies")
        feature_names.append(feature_name)
        feature_weight = st.sidebar.slider("Enter weight for feature #{feature_ndx}".format(feature_ndx=feature_ndx+1), 0.0, 1.0, step=0.1, value=1.0)
        feature_weights.append(feature_weight)

    dataframe = generate_synthetic_dataset(n_samples=n_samples, 
                                 n_classes=2, 
                                 class_weights=[class_weight, round(1.0-class_weight,1)], 
                                #  class_names=class_names,
                                 n_features=n_features, 
                                 feature_names=feature_names,
                                 feature_weights=feature_weights
                                 )

    st.dataframe(dataframe, width=1000, height=1000)
