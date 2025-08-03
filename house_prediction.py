import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from mlflow.models import infer_signature
from urllib.parse import urlparse

housing=fetch_california_housing()
#print(housing)
data=pd.DataFrame(housing.data,columns=housing.feature_names) #Since originally data was in form of arrays
data["Price"]=housing.target
##Independent and Dependent features
X=data.drop(columns=["Price"])
y=data["Price"]
#print(data)

#Split train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from mlflow.models import infer_signature #infer signature is used to set schema based on input and output
signature=infer_signature(X_train,y_train)

#Hyperparameter Tuning using GridSearch
def hyperparameter_tuning(X_train,y_train,paramgrid):
    rf=RandomForestRegressor()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2,
                             scoring="neg_mean_squared_error")
    grid_search.fit(X_train,y_train)
    return grid_search

#Define Hyperparameter Grid
param_grid={'n_estimators':[100,200],'max_depth':[5,10,None],
            'min_samples_split':[2,5], 'min_samples_leaf':[1,2]}

#Start Mlflow Experiment
with mlflow.start_run():
    #Perform hyperparameter Tuning
    grid_search=hyperparameter_tuning(X_train,y_train,param_grid)

    # Get Best Model
    best_model=grid_search.best_estimator_

    #Evaluate Best Model

    y_pred=best_model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)

    #Log best parameters
    mlflow.log_params({
        "best_n_estimators": grid_search.best_params_["n_estimators"],
        "best_max_depth": grid_search.best_params_["max_depth"],
        "best_min_samples_split": grid_search.best_params_["min_samples_split"],
        "best_min_samples_leaf": grid_search.best_params_["min_samples_leaf"]
    })

    #Log metric
    mlflow.log_metric("mse",mse)

    #Tracking Uri
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store !='file':
        mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best RandomForest Model")
    else:
        mlflow.sklearn.log_model(best_model,"model",signature=signature)


    print(f"Best Parameters are {grid_search.best_params_}")
    print(f"Mean Squared Error is {mse}")





