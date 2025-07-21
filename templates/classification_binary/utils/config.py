# General info
structure_type = 'single_file' # 'single_file' or 'multi_file'
file_one_path = '../data/course.csv' # Training or all
file_two_path = None # Testing or None
label = 'CourseCompletion'
primary_metric = 'f1'



# Preprocessing
drop_features = [ # Specify which features should be dropped from the dataframe
    label,
    'UserID',
    'CompletionRate'
]
value_mappings = { # Specify which features should have their values mapped, and to what
    # 'Feature': {'yes': True, 'no': False},
}
type_coercion = { # Specify which data type is expected for each feature
    # 'Age': 'int',
}
missing_handling = { # Specify how missing data should be handled for each feature
    # 'Age': 'median',
}



# Encoding
numerical_cols = [ # Categories to StandardScalar()

]
onehot_cols = [ # Categories that don't have any order
    "CourseCategory"
]
ordinal_cols = [ # Categories that do have an order
    
]



# Hyperparameter search
model_configs = {
    'dummy_classifier': {
        'search_type': None,
        'baseline': True,
        'param_grid': {}
    },

    'logistic_regression': {
        'search_type': 'grid',
        'baseline': True,
        'param_grid': {
            'model__C': [0.01, 0.1, 1, 10], # Inverse of regularization strength
            'model__penalty': ['l2'], # Which loss type is applied
            'model__solver': ['lbfgs'] # Algorithm used for optimization
        }
    },

    'random_forest': {
        'search_type': 'random',
        'baseline': False,
        'param_grid': {
            'model__max_depth': [5, 6, 8, 10, None], # Maximum node depth each tree can have
            'model__min_samples_split': [10, 50, 0.01, 0.02], # Minimum amount of examples required in a node to allow a split. A float represents a fraction
            'model__min_samples_leaf': [4, 8, 10, 20], # Minimum amount of examples that a leaf can have. A float represents a fraction
            'model__max_leaf_nodes': [20, 40, None], # Maximum amount of leaves each tree can have
            'model__n_estimators': [50, 100, 150], # The number of trees that are voting
            'model__max_samples': [0.15, 0.2], # The percent of examples that each tree trains on
            'model__max_features': ['sqrt'], # Maximum amount of features the model can consider at each split
            'model__criterion': ['gini'],
            'model__class_weight': ['balanced']
        },
        'n_iter': 75
    },

    'lightgbm': {
        'search_type': 'random',
        'baseline': False,
        'param_grid': {
            'model__max_depth': [3, 4, 6, 8, -1], # Maximum node depth each tree can have
            'model__n_estimators': [50, 100, 200, 350, 500], # The number of trees that are voting
            'model__learning_rate': [0.01, 0.03, 0.05], # Shrinks the contribution of each tree
            'model__num_leaves': [20, 30, 50, 70], # Maximum amount of leaves each tree can have
            'model__min_child_samples': [20, 50, 100], # Minimum amount of examples that a leaf can have
            'model__min_split_gain': [0.1, 0.5, 1], # Minimum gain required to make a split
            'model__subsample': [0.6, 0.8], # The percent of examples that each tree trains on
            'model__colsample_bytree': [0.6, 0.8, 1], # Fraction of features sampled per tree
            'model__reg_alpha': [0.1, 1],
            'model__reg_lambda': [.5, 1, 2, 3, 5, 10],
            'model__class_weight': ['none', 'balanced']
        },
        'n_iter': 75
    }
}