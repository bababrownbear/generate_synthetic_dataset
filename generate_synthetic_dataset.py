# Heavily inspired by https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py

import pandas as pd
import numpy as np
import streamlit as st
import base64
import random
import json
from json import JSONEncoder
import pathlib
from datetime import datetime

class SyntheticDataModel:
    def __init__(self, num_samples, n_classes, class_weights, n_features, feature_names, feature_types, feature_number_range, class_names, y_column_name, feature_positive_class_ratio, feature_negative_class_ratio, dataset_name):
        self.num_samples=num_samples 
        self.n_classes=n_classes
        self.class_weights=class_weights 
        self.n_features=n_features
        self.feature_names=feature_names
        self.feature_types=feature_types
        self.feature_number_range=feature_number_range
        self.class_names=class_names
        self.y_column_name=y_column_name
        self.feature_positive_class_ratio=feature_positive_class_ratio
        self.feature_negative_class_ratio=feature_negative_class_ratio
        self.dataset_name=dataset_name

class SyntheticDataModelEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

def generate_synthetic_dataset(num_samples=100, 
                    n_classes=2, 
                    class_weights=[0.5, 0.5], 
                    n_features=2, 
                    feature_names=['EatsVeggies','MathTestScore'],
                    feature_types=['Boolean','Number'],
                    feature_number_range=[[],[75,100,25,75]],
                    class_names=[True,False], 
                    y_column_name="IsAmazing",
                    feature_positive_class_ratio=[0.7,0.8],
                    feature_negative_class_ratio=[0.1,0.1]):

    if num_samples is None:
        raise ValueError("num_samples cannot be None")
    if num_samples < 10:
        raise ValueError("num_samples should be greater than 10")
    if n_classes is None:
        raise ValueError("n_classes cannot be None")
    if n_classes < 2:
        raise ValueError("n_classes should be greater than or equal to 2")
    if n_features is None:
        raise ValueError("n_features cannot be None")
    if n_features < 1:
        raise ValueError("n_features should be greater than or equal to 1")
    if round(np.sum(class_weights)) != 1.0:
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


    X = np.zeros(shape=(num_samples, n_features))
    X = pd.DataFrame(data=X, columns=feature_names)

    y = np.zeros((num_samples, 1))
    y = pd.DataFrame(data=y, columns=[y_column_name])

    start = 0
    stop = 0
    for i in range(n_classes):
        num_samples_for_class = int(num_samples * class_weights[i])
        stop = stop + num_samples_for_class
        y[start:stop] = class_names[i] 
        start = stop

    for ndx, col in enumerate(X.columns,0):
        col_positive_weight = feature_positive_class_ratio[ndx]
        col_negative_weight = feature_negative_class_ratio[ndx]
        num_samples_for_positive_class = int(num_samples * class_weights[0])
        num_positive_towards_positive_class = int(num_samples_for_positive_class * col_positive_weight)
        num_positive_towards_negative_class = int(num_samples_for_positive_class * col_negative_weight)
        if feature_types[ndx] == "Boolean":
            X[X.columns[ndx]][0:num_positive_towards_positive_class] = True
            X[X.columns[ndx]][num_positive_towards_positive_class:num_samples_for_positive_class] = False

            X[X.columns[ndx]][num_samples_for_positive_class:num_samples_for_positive_class+num_positive_towards_negative_class] = True
            X[X.columns[ndx]][num_samples_for_positive_class+num_positive_towards_negative_class:] = False
        elif feature_types[ndx] == 'Number':
            min_number_range_positive_class,max_number_range_positive_class,min_number_range_negative_class,max_number_range_negative_class = feature_number_range[ndx]
            X[X.columns[ndx]][0:num_positive_towards_positive_class] = [random.randint(min_number_range_positive_class,max_number_range_positive_class) for _ in range(num_positive_towards_positive_class)]
            X[X.columns[ndx]][num_positive_towards_positive_class:num_samples_for_positive_class] = [random.randint(min_number_range_negative_class,max_number_range_negative_class) for _ in range(num_samples_for_positive_class-num_positive_towards_positive_class)]

            X[X.columns[ndx]][num_samples_for_positive_class:num_samples_for_positive_class+num_positive_towards_negative_class] = [random.randint(min_number_range_positive_class,max_number_range_positive_class) for _ in range(num_positive_towards_negative_class)]
            X[X.columns[ndx]][num_samples_for_positive_class+num_positive_towards_negative_class:] = [random.randint(min_number_range_negative_class,max_number_range_negative_class) for _ in range(num_samples-num_samples_for_positive_class-num_positive_towards_negative_class)]

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
    return f'<a href="data:file/csv;base64,{b64}" download="{datasetname}.csv">[Download dataset as CSV]</a>'

def get_dataset_model_link(data_model_as_json):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    b64 = base64.b64encode(
        data_model_as_json.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/json;base64,{b64}" download="dataset_model.json">[Download dataset model as JSON]</a>'

if __name__ == '__main__':

    useJson = st.sidebar.checkbox("Use JSON", value=False)

    if useJson:
        with open('dataset_model.json') as f:
            json_from_file = json.load(f)
            json_input = st.sidebar.text_area("JSON Input", value=json.dumps(json_from_file))
            json_from_input = json.loads(json_input)
            data_model = SyntheticDataModel(**json_from_input)

    dataset_name = data_model.dataset_name if useJson else 'WhoIsAmazing'
    dataset_name = st.sidebar.text_input('Enter dataset name', value=dataset_name)
    
    num_samples = data_model.num_samples if useJson else 1000
    num_samples = st.sidebar.slider(str.format('Number of samples'), 10, 1000000, step=10, value=num_samples)

    y_column_name = data_model.y_column_name if useJson else 'IsAmazing'
    y_column_name = st.sidebar.text_input('Enter column name for y', value=y_column_name)

    n_classes = data_model.n_classes if useJson else 2

    class_names = []

    positive_name = data_model.class_names[0] if useJson else 'True'
    class_name = st.sidebar.text_input("Enter name for the positive class",value=positive_name)
    class_names.append(class_name)

    negative_name = data_model.class_names[1] if useJson else 'False'
    class_name = st.sidebar.text_input("Enter name for the negative class",value=negative_name)
    class_names.append(class_name)

    class_weight = data_model.class_weights[0] if useJson else 0.5
    class_weight = st.sidebar.slider('Ratio of positive to negative classes', 0.0, 1.0, step=0.01, value=class_weight)
    
    n_features = data_model.n_features if useJson else 1
    n_features = st.sidebar.slider('Number of features', 1, 25, step=1, value=n_features)

    feature_names = []
    feature_types = []
    feature_number_range = []
    feature_positive_class_ratio = []
    feature_negative_class_ratio = []
    for feature_ndx in range(n_features):
        st.sidebar.text("Feature #{feature_ndx}".format(feature_ndx=feature_ndx+1))
        
        if useJson:
            useJson = (feature_ndx < len(data_model.feature_names)) & useJson
        
        feature_name = data_model.feature_names[feature_ndx] if useJson else "EatsVeggies"
        feature_name = st.sidebar.text_input("Enter feature name for feature #{feature_ndx}".format(feature_ndx=feature_ndx+1), value=feature_name)
        feature_names.append(feature_name)

        feature_type = ['Boolean','Number'].index(data_model.feature_types[feature_ndx]) if useJson else 0
        feature_type = st.sidebar.selectbox("Select type for feature #{feature_ndx}".format(feature_ndx=feature_ndx+1), options=['Boolean', 'Number'], index=feature_type)

        if feature_type == 'Number':
            min_number_range_positive_class,max_number_range_positive_class,min_number_range_negative_class,max_number_range_negative_class = data_model.feature_number_range[feature_ndx] if useJson else [0,0,0,0]
            min_number_range_positive_class = st.sidebar.number_input("Enter min number range for positive class:",value=min_number_range_positive_class)
            max_number_range_positive_class = st.sidebar.number_input("Enter max number range for positive class:",value=max_number_range_positive_class)
            min_number_range_negative_class = st.sidebar.number_input("Enter min number range for negative class:",value=min_number_range_negative_class)
            max_number_range_negative_class = st.sidebar.number_input("Enter max number range for negative class:",value=max_number_range_negative_class)

            feature_number_range.append([min_number_range_positive_class,max_number_range_positive_class,min_number_range_negative_class,max_number_range_negative_class])
        else:
            feature_number_range.append([])

        feature_types.append(feature_type)

        feature_class_ratio = data_model.feature_positive_class_ratio[feature_ndx] if useJson else 0.85
        feature_class_ratio = st.sidebar.slider("Enter positive ratio for feature #{feature_ndx} vs positive class".format(feature_ndx=feature_ndx+1), 0.0, 1.0, step=0.01, value=feature_class_ratio) 
        feature_positive_class_ratio.append(feature_class_ratio)

        feature_class_ratio = data_model.feature_negative_class_ratio[feature_ndx] if useJson else 0.05
        feature_class_ratio = st.sidebar.slider("Enter positive ratio for feature #{feature_ndx} vs negative class".format(feature_ndx=feature_ndx+1), 0.0, 1.0, step=0.01, value=feature_class_ratio)
        feature_negative_class_ratio.append(feature_class_ratio)

    dataframe = generate_synthetic_dataset(num_samples=num_samples, 
                                 n_classes=2, 
                                 class_weights=[class_weight, round(1.0-class_weight,1)], 
                                 class_names=class_names,
                                 y_column_name=y_column_name,
                                 n_features=n_features, 
                                 feature_names=feature_names,
                                 feature_types=feature_types,
                                 feature_number_range=feature_number_range,
                                 feature_positive_class_ratio=feature_positive_class_ratio,
                                 feature_negative_class_ratio=feature_negative_class_ratio
                                 )

    # st.dataframe(dataframe, width=1000, height=1000)

    st.markdown(get_table_download_link(dataframe, dataset_name), unsafe_allow_html=True)

    synthetic_data_model = SyntheticDataModel(num_samples=num_samples, 
                                 n_classes=2, 
                                 class_weights=[class_weight, round(1.0-class_weight,1)], 
                                 class_names=class_names,
                                 y_column_name=y_column_name,
                                 n_features=n_features, 
                                 feature_names=feature_names,
                                 feature_types=feature_types,
                                 feature_number_range=feature_number_range,
                                 feature_positive_class_ratio=feature_positive_class_ratio,
                                 feature_negative_class_ratio=feature_negative_class_ratio,
                                 dataset_name=dataset_name)

    syntheic_data_model_json = json.dumps(synthetic_data_model, indent=4, cls=SyntheticDataModelEncoder)

    st.markdown(get_dataset_model_link(syntheic_data_model_json), unsafe_allow_html=True)

    st.text(syntheic_data_model_json)

