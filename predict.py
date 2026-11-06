from pathlib import Path
import os
import logging
from joblib import load
from dotenv import load_dotenv

import pandas as pd


try:
        load_dotenv()
        PROJECT_ROOT=Path(os.getenv('PROJECT_ROOT'))
                
        MODEL_PATH=PROJECT_ROOT / os.getenv('MODELS')/os.getenv('MODEL_NAME')
        LOG_PATH=PROJECT_ROOT/os.getenv('LOG_DIR')/os.getenv('LOG_NAME')
                
                
        LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
                
        logging.basicConfig(
                            level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            handlers=[
                                logging.StreamHandler(),
                                logging.FileHandler(LOG_PATH)
                            ]
                        )
        
        logging.info("Prediction script strated ")

        loaded_model=load(MODEL_PATH)
        logging.info("Model loaded ")
        
        def predicted(input_data:dict):
            df=pd.DataFrame([input_data])
            predicted=loaded_model.predict(df)[0]
            return predicted
        
        # diabetes_input = {
        #          "Pregnancies": 6,
        #         "Glucose": 148,
        #         "BloodPressure": 72,
        #         "SkinThickness": 35,
        #         "Insulin": 0.0,
        #         "BMI": 33.6,
        #         "DiabetesPedigreeFunction": 0.627,
        #         "Age": 50
        #     }
        # prediction=predicted(diabetes_input)
        # if prediction == 1:
        #     print("Model prediction : diabetes")
        #     logging.info("Prediction result is : diabetes")
        # else :
        #     print("Model prediction : No diabetes")
        #     logging.info("Prediction result is : No diabetes")
            
        # logging.info("Prediction script finsihed ")
               
except Exception as e:
        print(f"Predction is failed {e}")
        logging.exception(" Predction script failed :",e)
        raise
 
 