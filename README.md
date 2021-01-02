Heavily inspired by sklearn's make_classification, this is a more granular widget for generating a dataset.

https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py

1. Using conda install environment.yml.
2. run 'streamlit run generate_synthetic_dataset.py
3. Or if you've saved a dataset model as JSON, download a dataset using 'python create_dataset_from_json.py dataset_model.json'

A new window in your browser should pop up, pointed to http://localhost:8501/.

View live on streamlit: https://share.streamlit.io/bababrownbear/generate_synthetic_dataset/generate_synthetic_dataset.py

Original version:
![image](https://user-images.githubusercontent.com/29419183/103145338-51074300-46fe-11eb-9409-72e9a9d5a73f.png)

Latest version:
![image](https://user-images.githubusercontent.com/29419183/103449399-87e4d800-4c6d-11eb-99f3-79de44e88d52.png)

Build and save your data model as json, to recreate your dataset and use "python create_dataset_from_json.py dataset_model.json"
