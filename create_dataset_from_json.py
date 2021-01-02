import generate_synthetic_dataset as gsd
import json
import sys

with open(f'{sys.argv[1]}') as f:
  json_from_file = json.load(f)
  data_model = gsd.SyntheticDataModel(**json_from_file)
  dataset = gsd.generate_synthetic_dataset(n_samples=data_model.n_samples, 
                                 n_classes=data_model.n_classes, 
                                 class_weights=data_model.class_weights, 
                                 class_names=data_model.class_names,
                                 y_column_name=data_model.y_column_name,
                                 n_features=data_model.n_features, 
                                 feature_names=data_model.feature_names,
                                 feature_types=data_model.feature_types,
                                 feature_number_range=data_model.feature_number_range,
                                 feature_positive_class_ratio=data_model.feature_positive_class_ratio,
                                 feature_negative_class_ratio=data_model.feature_negative_class_ratio)
  dataset.to_csv(f"{data_model.dataset_name}.csv", index=False)