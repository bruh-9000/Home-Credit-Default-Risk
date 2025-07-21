import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1] / 'templates' / 'classification_binary'
sys.path.append(str(ROOT))

from utils.utils import prepare_train_test_split



def test_prepare_train_test_split_multi_file():
    train_df = pd.DataFrame({
        'CourseCompletion': [0, 1],
        'UserID': [1, 2],
        'CompletionRate': [0.1, 0.2],
        'CourseCategory': ['A', 'B']
    })
    test_df = pd.DataFrame({
        'CourseCompletion': [1, 0],
        'UserID': [3, 4],
        'CompletionRate': [0.3, 0.4],
        'CourseCategory': ['A', 'B']
    })

    X, y, X_train, X_test, y_train, y_test = prepare_train_test_split(train_df, test_df, 'CourseCompletion')

    assert y_train.equals(train_df['CourseCompletion'])
    assert y_test.equals(test_df['CourseCompletion'])
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_pipeline_saving(tmp_path):
    pipe = Pipeline([('model', DummyClassifier())])
    path = tmp_path / 'pipe.pkl'
    joblib.dump(pipe, path)
    assert path.exists()


def test_pipeline_reload(tmp_path):
    pipe = Pipeline([('model', DummyClassifier())])
    path = tmp_path / 'pipe.pkl'
    joblib.dump(pipe, path)
    loaded = joblib.load(path)
    assert isinstance(loaded, Pipeline)