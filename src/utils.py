import os
import sys
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully at: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # Get the hyperparameters for the current model
            model_params = param.get(model_name, {})

            # Initialize GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=model_params,
                cv=3,
                n_jobs=-1,
                verbose=1,
                refit=True
            )

            # Fit the model with GridSearchCV
            gs.fit(X_train, y_train)

            # Set the best parameters found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict on training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report
            report[model_name] = test_model_score

            logging.info(f"Model {model_name} evaluation completed. Train Score: {train_model_score}, Test Score: {test_model_score}")

        return report
    except Exception as e:
        raise CustomException(e, sys)