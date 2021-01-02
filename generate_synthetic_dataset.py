# Heavily inspired by https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py

import pandas as pd
import numpy as np
import streamlit as st
import base64


def generate_synthetic_dataset(n_samples=10, 
                    n_classes=2, 
                    class_weights=[0.5, 0.5], 
                    n_features=1, 
                    feature_names=['EatsVeggies'],
                    feature_types=['Boolean'],
                    class_names=[True,False], 
                    y_column_name="IsAmazing",
                    feature_positive_class_ratio=[0.7],
                    feature_negative_class_ratio=[0.1]):

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
    if feature_positive_class_ratio is None:
        raise ValueError("feature_positive_class_ratio cannot be None")
    if len(feature_positive_class_ratio) != n_features:
        raise ValueError("n_features must equal length of feature_positive_class_ratio")
    feature_positive_class_ratio_array = np.asarray(feature_positive_class_ratio)
    if (feature_positive_class_ratio_array > 1.0).sum() > 0:
        raise ValueError('feature_class_ratio cannot be greater than 1.0')
    if feature_negative_class_ratio is None:
        raise ValueError("feature_negative_class_ratio cannot be None")
    if len(feature_negative_class_ratio) != n_features:
        raise ValueError("n_features must equal length of feature_negative_class_ratio")
    feature_negative_class_ratio_array = np.asarray(feature_negative_class_ratio)
    if (feature_negative_class_ratio_array > 1.0).sum() > 0:
        raise ValueError('feature_class_ratio cannot be greater than 1.0')


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
        col_positive_weight = feature_positive_class_ratio[ndx]
        col_negative_weight = feature_negative_class_ratio[ndx]
        n_samples_for_positive_class = int(n_samples * class_weights[0])
        num_positive_towards_positive_class = int(n_samples_for_positive_class * col_positive_weight)
        num_positive_towards_negative_class = int(n_samples_for_positive_class * col_negative_weight)
        if feature_types[ndx] == "Boolean":
            X[X.columns[ndx]][0:num_positive_towards_positive_class] = True
            X[X.columns[ndx]][num_positive_towards_positive_class:n_samples_for_positive_class+num_positive_towards_positive_class] = False

            X[X.columns[ndx]][n_samples_for_positive_class:n_samples_for_positive_class+num_positive_towards_negative_class] = True
            X[X.columns[ndx]][n_samples_for_positive_class+num_positive_towards_negative_class:] = False

    dataset = pd.DataFrame(data=pd.concat([X, y], axis=1))

    return dataset

def get_table_download_link(df,datasetname):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{datasetname}.csv">Download dataset as CSV file</a>'

if __name__ == '__main__':
    dataset_name = st.sidebar.text_input('Enter dataset name', value='WhoIsAmazing')
    
    n_samples = st.sidebar.slider(str.format('Number of samples'), 10, 100, step=10)

    class_weight = st.sidebar.slider('Ratio of classes', 0.0, 1.0, step=0.1, value=0.5)

    st.sidebar.text_input('Enter column name for y', value='IsAmazing')

    n_classes = 2

    class_names = []

    class_name = st.sidebar.text_input("Enter name for the positive class",value="True")
    class_names.append(class_name)

    class_name = st.sidebar.text_input("Enter name for the negative class",value="False")
    class_names.append(class_name)
    
    n_features = st.sidebar.slider('Number of features', 1, 10, step=1)

    feature_names = []
    feature_types = []
    feature_positive_class_ratio = []
    feature_negative_class_ratio = []
    for feature_ndx in range(n_features):
        feature_name = st.sidebar.text_input("Enter feature name for feature #{feature_ndx}".format(feature_ndx=feature_ndx+1), value="EatsVeggies")
        feature_names.append(feature_name)
        feature_type = st.sidebar.selectbox("Select type for feature #{feature_ndx}".format(feature_ndx=feature_ndx+1), options=['Boolean'])
        feature_types.append(feature_type)
        feature_class_ratio = st.sidebar.slider("Enter positive ratio for feature #{feature_ndx} vs positive class".format(feature_ndx=feature_ndx+1), 0.0, 1.0, step=0.1, value=0.5)
        feature_positive_class_ratio.append(feature_class_ratio)
        feature_class_ratio = st.sidebar.slider("Enter positive ratio for feature #{feature_ndx} vs negative class".format(feature_ndx=feature_ndx+1), 0.0, 1.0, step=0.1, value=0.5)
        feature_negative_class_ratio.append(feature_class_ratio)

    dataframe = generate_synthetic_dataset(n_samples=n_samples, 
                                 n_classes=2, 
                                 class_weights=[class_weight, round(1.0-class_weight,1)], 
                                 class_names=class_names,
                                 n_features=n_features, 
                                 feature_names=feature_names,
                                 feature_types=feature_types,
                                 feature_positive_class_ratio=feature_positive_class_ratio,
                                 feature_negative_class_ratio=feature_negative_class_ratio
                                 )

    st.dataframe(dataframe, width=1000, height=1000)

    st.markdown(get_table_download_link(dataframe, dataset_name), unsafe_allow_html=True)

