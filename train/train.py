import pandas as pd 
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def ingest_data():
    df = df[["survived","pclass","sex","age"]]
    return df

def clean_data():
    df = df.dropna(axis=0)
    df  
    return  df

def train_model():
    model = KNeighborsClassifier()
    y = df['survived']
    X =  df.drop('survived', axis =1)
    model.fit(X,y)
    model.score(X,y)
    X_train, X_test, y_train, y_test= train_test_split (X,y, test_size=0.20, random_state=40)
    return model

if __name__ = "__main__":
    df = ingest_data("titanic.xls")
    df = clean_data(df)
    model = train_model(df)
    joblib.load(model)