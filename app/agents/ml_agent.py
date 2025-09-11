""" Minimal Auto ML Agent - Works purely on user queries"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from app.utils.data_loader import load_data

logger = get_logger(__name__)

class MLAgent:
    """ Auto ML Agent - understands any query, uses any model"""
    def __init__(self):
        self.name = "MLAgent"
        self.models = {}
        self.preprocessors = {}
        logger.info(f"Initialized {self.name}")

    def _auto_install(self, package: str) -> bool:
        """Auto-install any package"""
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

    def _auto_import(self, module_path: str, class_name: str):
        """Auto-import any model, install if needed"""
        try:
            import importlib
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # Auto-install based on module
                packages = {
                    "xgboost": "xgboost",
                    "catboost": "catboost", 
                    "lightgbm": "lightgbm",
                    "sklearn": "scikit-learn"
                }
                pkg = packages.get(module_path.split('.')[0], module_path.split('.')[0])
                if self._auto_install(pkg):
                    module = importlib.import_module(module_path)
                else:
                    return None
            
            return getattr(module, class_name)
        except:
            return None

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute any ML task automatically"""
        try:
            query = task.inputs.get("query", "").lower()
            data_path = task.inputs.get("data_path")
            target = task.inputs.get("target_column")
            
            # Auto-detect intent and execute
            if any(word in query for word in ["train", "build", "create", "fit"]):
                return self._auto_train(query, data_path, target)
            elif any(word in query for word in ["predict", "forecast", "classify"]):
                return self._auto_predict(query, data_path)
            elif any(word in query for word in ["cluster", "group"]):
                return self._auto_cluster(query, data_path)
            elif any(word in query for word in ["pca", "reduce", "dimension", "compress"]):
                return self._auto_reduce_dimensions(query, data_path)
            else:
                return self._auto_ml(query, data_path, target)
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _auto_train(self, query: str, data_path: str, target: str = None) -> Dict[str, Any]:
        """Auto-train any model based on query"""
        try:
            # Load data
            df = load_data(data_path)
            if df is None:
                return {"status": "error", "errors": ["Could not load data"]}
            
            # Auto-detect target
            if not target:
                target = df.columns[-1]
            
            # Auto-detect task type
            if target not in df.columns:
                task_type = "clustering"
                X = df
                y = None
            else:
                y = df[target]
                X = df.drop(columns=[target])
                task_type = "classification" if y.dtype == 'object' or len(y.unique()) < 20 else "regression"
            
            # Auto-select model from query
            model = self._auto_select_model(query, task_type, len(df))
            if not model:
                return {"status": "error", "errors": ["Could not create model"]}
            
            # Auto-preprocess
            X_processed = self._auto_preprocess(X)
            if y is not None:
                y_processed = self._auto_preprocess_target(y)
            
            # Auto-train
            if task_type == "clustering":
                model.fit(X_processed)
                labels = getattr(model, 'labels_', model.predict(X_processed))
                result = {"cluster_labels": labels.tolist(), "n_clusters": len(np.unique(labels))}
            else:
                # Split and train
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                result = {"train_score": train_score, "test_score": test_score}
            
            # Save model
            model_id = f"model_{len(self.models)}"
            self.models[model_id] = model
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": type(model).__name__,
                "task_type": task_type,
                **result
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _auto_predict(self, query: str, data_path: str) -> Dict[str, Any]:
        """Auto-predict using latest model"""
        try:
            if not self.models:
                return {"status": "error", "errors": ["No trained model available"]}
            
            # Use latest model
            model_id = list(self.models.keys())[-1]
            model = self.models[model_id]
            
            # Load and preprocess data
            df = load_data(data_path)
            X_processed = self._auto_preprocess(df)
            
            # Predict
            predictions = model.predict(X_processed)
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_processed)
            
            return {
                "status": "success",
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist() if probabilities is not None else None,
                "model_used": type(model).__name__
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _auto_cluster(self, query: str, data_path: str) -> Dict[str, Any]:
        """Auto-cluster data"""
        try:
            df = load_data(data_path)
            X_processed = self._auto_preprocess(df)
            
            # Auto-select clustering model
            model = self._auto_select_model(query, "clustering", len(df))
            
            # Fit and get labels
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(X_processed)
            else:
                model.fit(X_processed)
                labels = model.labels_
            
            return {
                "status": "success",
                "cluster_labels": labels.tolist(),
                "n_clusters": len(np.unique(labels)),
                "model_type": type(model).__name__
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _auto_ml(self, query: str, data_path: str, target: str = None) -> Dict[str, Any]:
        """Auto ML pipeline"""
        try:
            df = load_data(data_path)
            
            # Try different approaches automatically
            results = []
            
            # Try as supervised learning
            if target and target in df.columns:
                result = self._auto_train(f"train {query}", data_path, target)
                if result["status"] == "success":
                    results.append(("supervised", result))
            
            # Try as clustering
            cluster_result = self._auto_cluster(f"cluster {query}", data_path)
            if cluster_result["status"] == "success":
                results.append(("clustering", cluster_result))
            
            if results:
                best_approach, best_result = results[0]  # Use first successful approach
                return {
                    "status": "success",
                    "approach": best_approach,
                    "result": best_result,
                    "message": f"Auto-selected {best_approach} approach"
                }
            else:
                return {"status": "error", "errors": ["Could not find suitable ML approach"]}
                
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _auto_select_model(self, query: str, task_type: str, n_samples: int):
        """Auto-select best model based on query and data"""
        
        # Model mappings for different libraries
        models = {
            # User-specified models
            "xgboost": ("xgboost", "XGBClassifier" if task_type == "classification" else "XGBRegressor"),
            "catboost": ("catboost", "CatBoostClassifier" if task_type == "classification" else "CatBoostRegressor"),
            "lightgbm": ("lightgbm", "LGBMClassifier" if task_type == "classification" else "LGBMRegressor"),
            "random forest": ("sklearn.ensemble", "RandomForestClassifier" if task_type == "classification" else "RandomForestRegressor"),
            "logistic": ("sklearn.linear_model", "LogisticRegression"),
            "linear": ("sklearn.linear_model", "LinearRegression"),
            "svm": ("sklearn.svm", "SVC" if task_type == "classification" else "SVR"),
            "kmeans": ("sklearn.cluster", "KMeans"),
            "dbscan": ("sklearn.cluster", "DBSCAN"),
            "pca": ("sklearn.decomposition", "PCA"),
            "tsne": ("sklearn.manifold", "TSNE")
        }
        
        # Check if user specified a model
        for model_name, (module, class_name) in models.items():
            if model_name in query:
                model_class = self._auto_import(module, class_name)
                if model_class:
                    return self._create_model(model_class, model_name)
        
        # Auto-select based on task and data size
        if task_type == "classification":
            if n_samples > 10000:
                model_class = self._auto_import("sklearn.ensemble", "GradientBoostingClassifier")
            else:
                model_class = self._auto_import("sklearn.ensemble", "RandomForestClassifier")
        elif task_type == "regression":
            if n_samples > 10000:
                model_class = self._auto_import("sklearn.ensemble", "GradientBoostingRegressor")
            else:
                model_class = self._auto_import("sklearn.ensemble", "RandomForestRegressor")
        else:  # clustering
            if "dbscan" in query:
                model_class = self._auto_import("sklearn.cluster", "DBSCAN")
            else:
                model_class = self._auto_import("sklearn.cluster", "KMeans")
        
        return self._create_model(model_class, "auto") if model_class else None
    
    def _auto_reduce_dimensions(self, query: str, data_path: str) -> Dict[str, Any]:
        """Auto-reduce dimensions using PCA or t-SNE"""
        try:
            df = load_data(data_path)
            X_processed = self._auto_preprocess(df)
            
            # Auto-select dimensionality reduction method
            if "tsne" in query.lower():
                model_class = self._auto_import("sklearn.manifold", "TSNE")
                model = model_class(n_components=2, random_state=42)
            else:  # Default to PCA
                model_class = self._auto_import("sklearn.decomposition", "PCA")
                # Extract number of components from query
                import re
                numbers = re.findall(r'\d+', query)
                n_components = int(numbers[0]) if numbers else 2
                model = model_class(n_components=n_components)
            
            # Apply dimensionality reduction
            transformed_data = model.fit_transform(X_processed)
            
            # Save model
            model_id = f"pca_model_{len(self.models)}"
            self.models[model_id] = model
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_type": type(model).__name__,
                "original_dimensions": X_processed.shape[1],
                "reduced_dimensions": transformed_data.shape[1],
                "transformed_data": transformed_data.tolist(),
                "explained_variance_ratio": model.explained_variance_ratio_.tolist() if hasattr(model, 'explained_variance_ratio_') else None
            }
            
        except Exception as e:
            return {"status": "error", "errors": [str(e)]}

    def _create_model(self, model_class, model_name: str):
        """Create model instance with smart defaults"""
        try:
            if "KMeans" in model_class.__name__:
                return model_class(n_clusters=3, random_state=42)
            elif "DBSCAN" in model_class.__name__:
                return model_class()
            elif "PCA" in model_class.__name__:
                return model_class(n_components=2)
            elif "TSNE" in model_class.__name__:
                return model_class(n_components=2, random_state=42)
            else:
                try:
                    return model_class(random_state=42)
                except:
                    return model_class()
        except:
            return None

    def _auto_preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Auto-preprocess features"""
        X_processed = X.copy()
        
        # Handle categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.preprocessors:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.preprocessors[col] = le
            else:
                X_processed[col] = self.preprocessors[col].transform(X_processed[col].astype(str))
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean(numeric_only=True))
        X_processed = X_processed.fillna(0)  # For any remaining NaN
        
        # Scale numeric columns
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0 and "scaler" not in self.preprocessors:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
            self.preprocessors["scaler"] = scaler
        elif "scaler" in self.preprocessors:
            X_processed[numeric_cols] = self.preprocessors["scaler"].transform(X_processed[numeric_cols])
        
        return X_processed.values

    def _auto_preprocess_target(self, y: pd.Series) -> np.ndarray:
        """Auto-preprocess target"""
        if y.dtype == 'object':
            if "target_encoder" not in self.preprocessors:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_processed = le.fit_transform(y)
                self.preprocessors["target_encoder"] = le
            else:
                y_processed = self.preprocessors["target_encoder"].transform(y)
            return y_processed
        return y.values