""" Model Evaluation Agent - Combined Evaluation and Optimization"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from app.utils.data_loader import load_data

logger = get_logger(__name__)

class ModelEvaluationAgent:
    """ Combined Model Evaluation and Optimization Agent"""
    def __init__(self):
        self.name = "ModelEvaluationAgent"
        self.models = {}
        self.optimization_history = {}
        logger.info(f"Initialized {self.name}")

    def _auto_install(self, package: str) -> bool:
        """Auto-install packages"""
        try:
            import importlib
            importlib.import_module(package)
            return True
        except ImportError:
            try:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                return True
            except:
                return False

    def _auto_import(self, module_path: str, class_name: str = ""):
        """Auto-import with installation"""
        try:
            import importlib
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                packages = {
                    "sklearn": "scikit-learn",
                    "optuna": "optuna",
                    "hyperopt": "hyperopt"
                }
                pkg = packages.get(module_path.split('.')[0], module_path.split('.')[0])
                if self._auto_install(pkg):
                    module = importlib.import_module(module_path)
                else:
                    return None
            return getattr(module, class_name) if class_name else module
        except:
            return None

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute evaluation or optimization tasks"""
        try:
            query = task.inputs.get("query", "").lower()
            
            # Intent detection
            if any(word in query for word in ["evaluate", "metrics", "score", "performance"]):
                return self._evaluate_model(query, task.inputs)
            elif any(word in query for word in ["optimize", "tune", "hyperparameter", "improve"]):
                return self._optimize_model(query, task.inputs)
            elif any(word in query for word in ["benchmark", "compare", "baseline"]):
                return self._benchmark_model(query, task.inputs)
            elif any(word in query for word in ["grid search", "random search", "bayesian"]):
                return self._hyperparameter_search(query, task.inputs)
            else:
                return self._auto_evaluate_optimize(query, task.inputs)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _evaluate_model(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            model = inputs.get("model")
            X_test = inputs.get("X_test")
            y_test = inputs.get("y_test")
            y_pred = inputs.get("y_pred")
            task_type = inputs.get("task_type", "classification")
            
            if not any([model, y_pred]):
                return {"status": "error", "errors": ["Need model or predictions"]}
            
            # Get predictions if not provided
            if y_pred is None and model and X_test is not None:
                y_pred = model.predict(X_test)
            
            # Calculate metrics based on task type
            if task_type == "classification":
                metrics = self._calculate_classification_metrics(y_test, y_pred)
            else:
                metrics = self._calculate_regression_metrics(y_test, y_pred)
            
            # Generate grade and recommendations
            grade = self._assign_grade(metrics)
            recommendations = self._generate_recommendations(metrics, query)
            
            return {
                "status": "success",
                "analysis_type": "Model Evaluation",
                "task_type": task_type,
                "metrics": metrics,
                "grade": grade,
                "recommendations": recommendations,
                "message": f"Model evaluation completed with grade: {grade}"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _optimize_model(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize model hyperparameters"""
        try:
            model_type = inputs.get("model_type", "RandomForest")
            X_train = inputs.get("X_train")
            y_train = inputs.get("y_train")
            X_val = inputs.get("X_val")
            y_val = inputs.get("y_val")
            
            if X_train is None or y_train is None:
                return {"status": "error", "errors": ["Need training data"]}
            
            # Choose optimization method
            if "bayesian" in query or "optuna" in query:
                result = self._bayesian_optimization(model_type, X_train, y_train, X_val, y_val)
            elif "grid" in query:
                result = self._grid_search_optimization(model_type, X_train, y_train)
            else:
                result = self._random_search_optimization(model_type, X_train, y_train)
            
            # Save optimization history
            opt_id = f"optimization_{len(self.optimization_history)}"
            self.optimization_history[opt_id] = result
            
            return {
                "status": "success",
                "analysis_type": "Model Optimization",
                "optimization_id": opt_id,
                "best_params": result["best_params"],
                "best_score": result["best_score"],
                "improvement": result.get("improvement", "N/A"),
                "method": result["method"],
                "message": f"Model optimized using {result['method']}"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _benchmark_model(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark model against baselines"""
        try:
            model_score = inputs.get("model_score", 0.85)
            task_type = inputs.get("task_type", "classification")
            
            # Define baselines
            if task_type == "classification":
                baselines = {
                    "random_baseline": 0.5,
                    "majority_class": inputs.get("majority_baseline", 0.6),
                    "simple_model": 0.75,
                    "previous_best": inputs.get("previous_best", 0.82)
                }
            else:
                baselines = {
                    "mean_baseline": inputs.get("mean_baseline", 100.0),
                    "simple_model": inputs.get("simple_baseline", 80.0),
                    "previous_best": inputs.get("previous_best", 75.0)
                }
            
            # Calculate improvements
            improvements = {}
            ranking = 1
            for name, score in baselines.items():
                if task_type == "classification":
                    improvement = ((model_score - score) / score) * 100
                else:
                    improvement = ((score - model_score) / score) * 100  # Lower is better for regression
                improvements[name] = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                if (task_type == "classification" and score >= model_score) or \
                   (task_type == "regression" and score <= model_score):
                    ranking += 1
            
            return {
                "status": "success",
                "analysis_type": "Model Benchmarking",
                "model_score": model_score,
                "baselines": baselines,
                "improvements": improvements,
                "ranking": f"{ranking} out of {len(baselines) + 1} models",
                "message": f"Model ranks #{ranking} among {len(baselines) + 1} models"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _hyperparameter_search(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hyperparameter search"""
        try:
            search_type = "random"
            if "grid" in query:
                search_type = "grid"
            elif "bayesian" in query:
                search_type = "bayesian"
            
            model_type = inputs.get("model_type", "RandomForest")
            X_train = inputs.get("X_train")
            y_train = inputs.get("y_train")
            
            # Get parameter space
            param_space = self._get_parameter_space(model_type)
            
            if search_type == "grid":
                result = self._grid_search_optimization(model_type, X_train, y_train)
            elif search_type == "bayesian":
                result = self._bayesian_optimization(model_type, X_train, y_train)
            else:
                result = self._random_search_optimization(model_type, X_train, y_train)
            
            return {
                "status": "success",
                "analysis_type": f"{search_type.title()} Search",
                "search_type": search_type,
                "parameter_space": param_space,
                "best_params": result["best_params"],
                "best_score": result["best_score"],
                "trials_completed": result.get("n_trials", "N/A"),
                "message": f"{search_type.title()} search completed"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _calculate_classification_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate classification metrics"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add AUC if binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics["auc_roc"] = roc_auc_score(y_true, y_pred)
                except:
                    pass
            
            return {k: round(v, 4) for k, v in metrics.items()}
            
        except Exception as e:
            return {"accuracy": 0.0, "error": str(e)}

    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics"""
        try:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics = {
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2_score": r2_score(y_true, y_pred)
            }
            
            return {k: round(v, 4) for k, v in metrics.items()}
            
        except Exception as e:
            return {"mse": float('inf'), "error": str(e)}

    def _grid_search_optimization(self, model_type: str, X_train, y_train) -> Dict[str, Any]:
        """Grid search optimization"""
        try:
            GridSearchCV = self._auto_import("sklearn.model_selection", "GridSearchCV")
            if not GridSearchCV:
                return {"method": "grid_search", "error": "Could not import GridSearchCV"}
            
            # Get model and parameters
            model, param_grid = self._get_model_and_params(model_type)
            
            # Perform grid search
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            return {
                "method": "grid_search",
                "best_params": grid_search.best_params_,
                "best_score": round(grid_search.best_score_, 4),
                "n_trials": len(grid_search.cv_results_['params'])
            }
            
        except Exception as e:
            return {"method": "grid_search", "error": str(e)}

    def _random_search_optimization(self, model_type: str, X_train, y_train) -> Dict[str, Any]:
        """Random search optimization"""
        try:
            RandomizedSearchCV = self._auto_import("sklearn.model_selection", "RandomizedSearchCV")
            if not RandomizedSearchCV:
                return {"method": "random_search", "error": "Could not import RandomizedSearchCV"}
            
            # Get model and parameters
            model, param_dist = self._get_model_and_params(model_type, random=True)
            
            # Perform random search
            random_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1)
            random_search.fit(X_train, y_train)
            
            return {
                "method": "random_search",
                "best_params": random_search.best_params_,
                "best_score": round(random_search.best_score_, 4),
                "n_trials": 20
            }
            
        except Exception as e:
            return {"method": "random_search", "error": str(e)}

    def _bayesian_optimization(self, model_type: str, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        """Bayesian optimization using Optuna"""
        try:
            optuna = self._auto_import("optuna")
            if not optuna:
                return {"method": "bayesian", "error": "Could not import Optuna"}
            
            def objective(trial):
                # Get model with trial parameters
                model = self._get_model_with_trial_params(model_type, trial)
                
                # Train and evaluate
                model.fit(X_train, y_train)
                if X_val is not None and y_val is not None:
                    score = model.score(X_val, y_val)
                else:
                    from sklearn.model_selection import cross_val_score
                    scores = cross_val_score(model, X_train, y_train, cv=3)
                    score = scores.mean()
                
                return score
            
            # Create study and optimize
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50, show_progress_bar=False)
            
            return {
                "method": "bayesian",
                "best_params": study.best_params,
                "best_score": round(study.best_value, 4),
                "n_trials": 50
            }
            
        except Exception as e:
            return {"method": "bayesian", "error": str(e)}

    def _get_model_and_params(self, model_type: str, random: bool = False):
        """Get model and parameter space"""
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=42)
            if random:
                from scipy.stats import randint
                params = {
                    'n_estimators': randint(10, 200),
                    'max_depth': randint(3, 20),
                    'min_samples_split': randint(2, 20)
                }
            else:
                params = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=42)
            if random:
                from scipy.stats import uniform
                params = {
                    'C': uniform(0.1, 10),
                    'max_iter': randint(100, 1000)
                }
            else:
                params = {
                    'C': [0.1, 1, 10],
                    'max_iter': [100, 500, 1000]
                }
        
        return model, params

    def _get_model_with_trial_params(self, model_type: str, trial):
        """Get model with Optuna trial parameters"""
        if model_type == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 10, 200),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=trial.suggest_float('C', 0.1, 10),
                max_iter=trial.suggest_int('max_iter', 100, 1000),
                random_state=42
            )

    def _get_parameter_space(self, model_type: str) -> Dict[str, Any]:
        """Get parameter space description"""
        if model_type == "RandomForest":
            return {
                "n_estimators": "Number of trees (10-200)",
                "max_depth": "Maximum tree depth (3-20)",
                "min_samples_split": "Minimum samples to split (2-20)"
            }
        else:
            return {
                "C": "Regularization strength (0.1-10)",
                "max_iter": "Maximum iterations (100-1000)"
            }

    def _assign_grade(self, metrics: Dict[str, float]) -> str:
        """Assign letter grade based on metrics"""
        if "accuracy" in metrics:
            score = metrics["accuracy"]
        elif "r2_score" in metrics:
            score = metrics["r2_score"]
        else:
            score = 0.5
        
        if score >= 0.95: return "A+"
        elif score >= 0.90: return "A"
        elif score >= 0.85: return "B+"
        elif score >= 0.80: return "B"
        elif score >= 0.75: return "C+"
        elif score >= 0.70: return "C"
        else: return "D"

    def _generate_recommendations(self, metrics: Dict[str, float], query: str) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if "accuracy" in metrics and metrics["accuracy"] < 0.8:
            recommendations.append("Consider feature engineering or more complex models")
        
        if "f1_score" in metrics and metrics["f1_score"] < 0.7:
            recommendations.append("Address class imbalance with sampling techniques")
        
        if "improve" in query:
            recommendations.append("Try hyperparameter tuning for better performance")
        
        if not recommendations:
            recommendations.append("Model performance is good, consider ensemble methods for further improvement")
        
        return recommendations

    def _auto_evaluate_optimize(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Auto evaluate and optimize"""
        try:
            # First evaluate
            eval_result = self._evaluate_model(query, inputs)
            
            # Then optimize if performance is low
            if eval_result.get("status") == "success":
                grade = eval_result.get("grade", "D")
                if grade in ["C", "D"]:
                    opt_result = self._optimize_model(query, inputs)
                    return {
                        "status": "success",
                        "analysis_type": "Auto Evaluate & Optimize",
                        "evaluation": eval_result,
                        "optimization": opt_result,
                        "message": "Completed evaluation and optimization"
                    }
            
            return eval_result
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}