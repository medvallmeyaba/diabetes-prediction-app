from pathlib import Path
import os
import logging
from joblib import dump
from dotenv import load_dotenv


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report


def train_model():
    try:
        load_dotenv()
        PROJECT_ROOT=Path(os.getenv('PROJECT_ROOT'))
        
        DATESET_PATH= PROJECT_ROOT / os.getenv('DATASET_NAME')
        MODEL_PATH=PROJECT_ROOT / os.getenv('MODELS')/os.getenv('MODEL_NAME')
        LOG_PATH=PROJECT_ROOT/os.getenv('LOG_DIR')/os.getenv('LOG_NAME')
        
        TARGET_COL=os.getenv("TAREGT_COL")
        TEST_SIZE=float(os.getenv("TEST_SIZE"))
        RANDOM_STATE=int(os.getenv("RANDOM_STATE"))
        
        MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
        
        logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler(LOG_PATH)
                    ]
                )
    
        logging.info("Trainnig script started ")
        df=pd.read_csv(DATESET_PATH)  
        logging.info(f"dataset shape : {df.shape}")        
        # Data preprocessing
        df_copy=df.drop(columns=[TARGET_COL])
        corr=df_copy.corr().abs()
        to_drop=set()

        for i in range(len(corr.columns)):
          for j in range(i):
            if corr.iloc[i,j] > 0.95:
               to_drop.add(corr.columns[i])
            
        select_features=[c for c in df_copy.columns if c not in to_drop]
        print(to_drop)
        print("*"*40)
        print(select_features)
        
        X=df.drop(columns=[TARGET_COL])  
        Y=df[TARGET_COL]
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        logging.info("Before SMOTE:", df.value_counts(y_train))

        smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
        x_train, y_train = smote.fit_resample(x_train, y_train)
        logging.info("After SMOTE:", df.value_counts(y_train))
        scaler=StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled  = scaler.transform(x_test)
        
        models={
        "LogisticRegression":LogisticRegression(max_iter=1000),
        "SVM":SVC(),
        "KNN":KNeighborsClassifier(n_neighbors=3),
        "DecisionTree":DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest":RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoostingRegressor":GradientBoostingClassifier(random_state=RANDOM_STATE)
       }
        
        scores={}
        for name,model in models.items():
           score=cross_val_score(model,x_train,y_train,cv=5)
           scores[name]=score.mean()
        
        best_model=max(scores,key=scores.get)
        logging.info(f"best model is : {best_model}")
        
#         param_grid = {
#                "RandomForest":{
#                             'model__n_estimators': [50, 100, 200],
#                             'model__max_depth': [3, 5, 7, None],
#                             'model__max_features': ['sqrt', 'log2', None],
#                             'model__min_samples_split': [2, 5, 10],
#                             'model__min_samples_leaf': [1, 2, 4],
#                             }
#                 }
#   # scaling  + model 
        pipe_line = Pipeline([
                ('scaler', StandardScaler()),
                ('model', models[best_model])
            ])

 # GridSearchCV automates the hyperparameter tuning process and helps to find the best hyperparameters
        # grid_search = GridSearchCV(
        #         estimator=pipe_line,
        #         param_grid=param_grid[best_model],
        #         cv=5,
        #         scoring='accuracy',
        #         n_jobs=1
        #     )
              
        pipe_line.fit(x_train, y_train)
        logging.info("Model trainning completed ")
        
        acc_train=accuracy_score(y_train,pipe_line.predict(x_train))
        acc_test=accuracy_score(y_test,pipe_line.predict(x_test))
        
        train_report=classification_report(y_train,pipe_line.predict(x_train))
        test_report=classification_report(y_test,pipe_line.predict(x_test))
        
        logging.info(f"Train Accuray {acc_train:.3f}")
        logging.info(f"Test Accuray {acc_test:.3f}")
        
        logging.info(f"Train classification_report \n{train_report}")
        logging.info(f"Test classification_report \n{test_report}")


        dump(pipe_line,MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")
        logging.info("Trainning script finished")
        


       
    except Exception as e:
        print(f"Trainning is failed {e}")
        logging.exception("Trainning script failed :",e)
        raise
 
 
if __name__== "__main__":
    train_model()    