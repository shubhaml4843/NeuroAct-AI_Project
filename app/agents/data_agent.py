""" Conversational Data Agent - Works like ChatGPT for data processing tasks"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from app.utils.data_loader import load_data
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

class DataAgent:
    """Conversational agent for data wrangling and processing tasks."""
    def __init__(self):
        self.name = "DataAgent"
        self.tools = ["pandas", "numpy", "sklearn"]
        self.allowed_dirs = ["data", "datasets", "uploads"]
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
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
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                return True
            except Exception as e:
                logger.warning(f"Could not install {package}: {str(e)}")
                return False
    
    def _auto_import(self, module_path: str, class_name: str):
        """Auto-import any data processing tool, install if needed"""
        try:
            import importlib
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # Auto-install based on module
                packages = {
                    "sklearn": "scikit-learn",
                    "seaborn": "seaborn",
                    "plotly": "plotly",
                    "scipy": "scipy",
                    "statsmodels": "statsmodels"
                }
                pkg = packages.get(module_path.split('.')[0], module_path.split('.')[0])
                if self._auto_install(pkg):
                    module = importlib.import_module(module_path)
                else:
                    return None
            
            return getattr(module, class_name)
        except Exception as e:
            logger.warning(f"Could not import {class_name} from {module_path}: {str(e)}")
            return None

    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        if not file_path:
            return False
        
        try:
            path = Path(file_path).resolve()
            is_allowed = any(allowed_dir in str(path) for allowed_dir in self.allowed_dirs)
            return is_allowed
        except Exception:
            return False

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute data processing based on natural language query"""
        logger.info(f"{self.name} executing task: {task.task_id}")
        try:
            query = task.inputs.get("query", "")
            return self._understand_and_execute(query, task.inputs)
        except Exception as e:
            logger.error(f"{self.name} execution failed: {str(e)}")
            return {"status": "error", "errors": [str(e)], "agent": self.name}

    def _understand_and_execute(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Understand natural language query and execute appropriate data task"""
        query_lower = query.lower()
        
        # Analyze what user wants to do
        intent = self._analyze_data_intent(query_lower)
        
        # Execute based on intent
        if intent == "clean":
            return self._smart_clean(query, inputs)
        elif intent == "analyze":
            return self._smart_analyze(query, inputs)
        elif intent == "transform":
            return self._smart_transform(query, inputs)
        elif intent == "load":
            return self._smart_load(query, inputs)
        elif intent == "export":
            return self._smart_export(query, inputs)
        elif intent == "eda":
            return self._smart_eda(query, inputs)
        elif intent == "pipeline":
            return self._smart_pipeline(query, inputs)
        else:
            return self._smart_general(query, inputs)
    
    def _analyze_data_intent(self, query: str) -> str:
        """Analyze what data operation user wants"""
        
        # Cleaning intents
        clean_keywords = ["clean", "remove duplicates", "handle missing", "fill na", "outliers", "prepare"]
        if any(keyword in query for keyword in clean_keywords):
            return "clean"
        
        # Analysis intents
        analyze_keywords = ["analyze", "statistics", "summary", "describe", "insights", "patterns", "correlations"]
        if any(keyword in query for keyword in analyze_keywords):
            return "analyze"
        
        # EDA intents
        eda_keywords = ["eda", "explore", "comprehensive", "distribution", "visualization"]
        if any(keyword in query for keyword in eda_keywords):
            return "eda"
        
        # Transform intents
        transform_keywords = ["transform", "scale", "encode", "normalize", "feature", "preprocess"]
        if any(keyword in query for keyword in transform_keywords):
            return "transform"
        
        # Load intents
        load_keywords = ["load", "read", "import", "open"]
        if any(keyword in query for keyword in load_keywords):
            return "load"
        
        # Export intents
        export_keywords = ["export", "save", "write", "output"]
        if any(keyword in query for keyword in export_keywords):
            return "export"
        
        # Pipeline intents
        pipeline_keywords = ["pipeline", "full process", "end to end", "complete"]
        if any(keyword in query for keyword in pipeline_keywords):
            return "pipeline"
        
        return "general"
    
    def _smart_clean(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart data cleaning based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = load_data(file_path)
            if df is None:
                return self._response("error", "clean", errors=["Could not load data"])
            
            original_rows = len(df)
            cleaning_operations = []
            
            # Determine cleaning operations based on query
            operations = self._extract_cleaning_operations(query)
            
            # Remove duplicates
            if "duplicates" in operations or "duplicate" in query.lower():
                before_dup = len(df)
                df.drop_duplicates(inplace=True)
                duplicates_removed = before_dup - len(df)
                if duplicates_removed > 0:
                    cleaning_operations.append(f"Removed {duplicates_removed} duplicate rows")
            
            # Handle missing values
            if "missing" in operations or "na" in query.lower() or "null" in query.lower():
                missing_before = df.isnull().sum().sum()
                
                if "advanced" in query.lower():
                    # Advanced imputation
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    
                    if len(numeric_cols) > 0:
                        imputer_class = self._auto_import("sklearn.impute", "SimpleImputer")
                        if imputer_class:
                            imputer_num = imputer_class(strategy='median')
                            df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
                    
                    if len(categorical_cols) > 0:
                        imputer_class = self._auto_import("sklearn.impute", "SimpleImputer")
                        if imputer_class:
                            imputer_cat = imputer_class(strategy='most_frequent')
                            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
                else:
                    # Basic imputation
                    for col in df.columns:
                        if df[col].isnull().any():
                            if df[col].dtype in [np.float64, np.int64]:
                                df[col].fillna(df[col].mean(), inplace=True)
                            else:
                                mode_val = df[col].mode()
                                if len(mode_val) > 0:
                                    df[col].fillna(mode_val[0], inplace=True)
                
                missing_after = df.isnull().sum().sum()
                missing_filled = missing_before - missing_after
                if missing_filled > 0:
                    cleaning_operations.append(f"Filled {missing_filled} missing values")
            
            # Handle outliers
            if "outlier" in query.lower():
                outliers_removed = 0
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    before_outlier = len(df)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    outliers_removed += before_outlier - len(df)
                
                if outliers_removed > 0:
                    cleaning_operations.append(f"Removed {outliers_removed} outliers")
            
            # Save if requested
            output_path = inputs.get("output_path")
            if output_path:
                df.to_csv(output_path, index=False)
                cleaning_operations.append(f"Saved to {output_path}")
            
            return self._response("success", "clean", {
                "message": f"Data cleaning completed: {', '.join(cleaning_operations)}",
                "original_rows": original_rows,
                "final_rows": len(df),
                "operations_performed": cleaning_operations,
                "data_shape": df.shape
            })
            
        except Exception as e:
            return self._response("error", "clean", errors=[f"Cleaning failed: {str(e)}"])
    
    def _smart_analyze(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart data analysis based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = load_data(file_path)
            if df is None:
                return self._response("error", "analyze", errors=["Could not load data"])
            
            analysis_results = {}
            
            # Basic statistics
            analysis_results["basic_stats"] = {
                "rows": len(df),
                "columns": len(df.columns),
                "missing_values": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            # Column analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            analysis_results["column_info"] = {
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "numeric_names": list(numeric_cols),
                "categorical_names": list(categorical_cols)
            }
            
            # Statistical summary for numeric columns
            if len(numeric_cols) > 0:
                analysis_results["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
            # Categorical analysis
            if len(categorical_cols) > 0:
                cat_analysis = {}
                for col in categorical_cols[:5]:
                    cat_analysis[col] = {
                        "unique_values": int(df[col].nunique()),
                        "top_values": df[col].value_counts().head(3).to_dict()
                    }
                analysis_results["categorical_summary"] = cat_analysis
            
            # Correlations if requested
            if "correlation" in query.lower() and len(numeric_cols) > 1:
                analysis_results["correlations"] = df[numeric_cols].corr().to_dict()
            
            # Generate insights
            insights = self._generate_data_insights(df)
            analysis_results["insights"] = insights
            
            return self._response("success", "analyze", {
                "message": f"Analysis completed for dataset with {len(df)} rows and {len(df.columns)} columns",
                "analysis": analysis_results
            })
            
        except Exception as e:
            return self._response("error", "analyze", errors=[f"Analysis failed: {str(e)}"])
    
    def _smart_transform(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart data transformation based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = load_data(file_path)
            if df is None:
                return self._response("error", "transform", errors=["Could not load data"])
            
            transformations_applied = []
            
            # Scale/normalize numeric features
            if any(word in query.lower() for word in ["scale", "normalize", "standardize"]):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    if "minmax" in query.lower():
                        scaler_class = self._auto_import("sklearn.preprocessing", "MinMaxScaler")
                        if scaler_class:
                            scaler = scaler_class()
                            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                            transformations_applied.append("MinMax scaling applied to numeric columns")
                    elif "robust" in query.lower():
                        scaler_class = self._auto_import("sklearn.preprocessing", "RobustScaler")
                        if scaler_class:
                            scaler = scaler_class()
                            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                            transformations_applied.append("Robust scaling applied to numeric columns")
                    else:
                        scaler_class = self._auto_import("sklearn.preprocessing", "StandardScaler")
                        if scaler_class:
                            scaler = scaler_class()
                            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                            transformations_applied.append("Standard scaling applied to numeric columns")
            
            # Encode categorical features
            if any(word in query.lower() for word in ["encode", "categorical"]):
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if df[col].nunique() <= 5:  # One-hot encode low cardinality
                        if "onehot" in query.lower():
                            encoder_class = self._auto_import("sklearn.preprocessing", "OneHotEncoder")
                            if encoder_class:
                                encoded = pd.get_dummies(df[col], prefix=col)
                                df = pd.concat([df.drop(col, axis=1), encoded], axis=1)
                                transformations_applied.append(f"One-hot encoded {col}")
                    else:  # Label encode high cardinality
                        encoder_class = self._auto_import("sklearn.preprocessing", "LabelEncoder")
                        if encoder_class:
                            le = encoder_class()
                            df[col] = le.fit_transform(df[col].astype(str))
                            transformations_applied.append(f"Label encoded {col}")
            
            # Create features from datetime
            if "datetime" in query.lower() or "date" in query.lower():
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                for col in datetime_cols:
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    transformations_applied.append(f"Created date features from {col}")
            
            # Save if requested
            output_path = inputs.get("output_path")
            if output_path:
                df.to_csv(output_path, index=False)
                transformations_applied.append(f"Saved to {output_path}")
            
            return self._response("success", "transform", {
                "message": f"Transformations completed: {', '.join(transformations_applied)}",
                "transformations": transformations_applied,
                "final_shape": df.shape,
                "new_columns": len(df.columns)
            })
            
        except Exception as e:
            return self._response("error", "transform", errors=[f"Transformation failed: {str(e)}"])
    
    def _smart_load(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart data loading based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = load_data(file_path)
            if df is None:
                return self._response("error", "load", errors=["Could not load data"])
            
            # Basic info about loaded data
            info = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
            }
            
            # Quick quality check
            quality_info = {
                "missing_values": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(df.select_dtypes(include=['object']).columns)
            }
            
            return self._response("success", "load", {
                "message": f"Successfully loaded {len(df)} rows × {len(df.columns)} columns",
                "data_info": info,
                "quality_check": quality_info
            })
            
        except Exception as e:
            return self._response("error", "load", errors=[f"Loading failed: {str(e)}"])
    
    def _smart_export(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart data export based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            output_path = inputs.get("output_path")
            
            if not output_path:
                return self._response("error", "export", errors=["No output path provided"])
            
            df = load_data(file_path)
            if df is None:
                return self._response("error", "export", errors=["Could not load data"])
            
            # Determine format from query or file extension
            if "json" in query.lower() or output_path.endswith('.json'):
                df.to_json(output_path, orient='records')
                format_used = "JSON"
            elif "parquet" in query.lower() or output_path.endswith('.parquet'):
                df.to_parquet(output_path)
                format_used = "Parquet"
            else:
                df.to_csv(output_path, index=False)
                format_used = "CSV"
            
            file_size = round(Path(output_path).stat().st_size / 1024 / 1024, 2)
            
            return self._response("success", "export", {
                "message": f"Data exported to {format_used} format",
                "output_path": output_path,
                "format": format_used,
                "rows_exported": len(df),
                "file_size_mb": file_size
            })
            
        except Exception as e:
            return self._response("error", "export", errors=[f"Export failed: {str(e)}"])
    
    def _smart_eda(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart exploratory data analysis based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = load_data(file_path)
            if df is None:
                return self._response("error", "eda", errors=["Could not load data"])
            
            eda_results = {}
            
            # Data profile
            eda_results["data_profile"] = {
                "shape": df.shape,
                "column_types": {
                    "numeric": len(df.select_dtypes(include=[np.number]).columns),
                    "categorical": len(df.select_dtypes(include=['object']).columns),
                    "datetime": len(df.select_dtypes(include=['datetime64']).columns)
                }
            }
            
            # Missing value analysis
            missing_info = df.isnull().sum()
            eda_results["missing_analysis"] = {
                "columns_with_missing": missing_info[missing_info > 0].to_dict(),
                "missing_percentage": (missing_info / len(df) * 100).to_dict()
            }
            
            # Correlation analysis for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlations = df[numeric_cols].corr()
                # Find high correlations
                high_corr = []
                for i in range(len(correlations.columns)):
                    for j in range(i+1, len(correlations.columns)):
                        corr_val = correlations.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr.append({
                                "feature1": correlations.columns[i],
                                "feature2": correlations.columns[j],
                                "correlation": round(corr_val, 3)
                            })
                
                eda_results["correlation_analysis"] = {
                    "correlation_matrix": correlations.to_dict(),
                    "high_correlations": high_corr
                }
            
            # Distribution analysis
            if len(numeric_cols) > 0:
                distribution_stats = {}
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    distribution_stats[col] = {
                        "skewness": float(df[col].skew()),
                        "kurtosis": float(df[col].kurtosis()),
                        "outliers": len(df[(df[col] < df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))) | 
                                         (df[col] > df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))])
                    }
                eda_results["distribution_analysis"] = distribution_stats
            
            # Data quality score
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
            quality_score = (completeness + uniqueness) / 2
            
            eda_results["data_quality"] = {
                "completeness_score": round(completeness, 2),
                "uniqueness_score": round(uniqueness, 2),
                "overall_quality_score": round(quality_score, 2)
            }
            
            # Generate insights and recommendations
            insights = self._generate_eda_insights(df, eda_results)
            recommendations = self._generate_eda_recommendations(df, eda_results)
            
            return self._response("success", "eda", {
                "message": f"Comprehensive EDA completed for {len(df)} rows × {len(df.columns)} columns",
                "eda_results": eda_results,
                "insights": insights,
                "recommendations": recommendations
            })
            
        except Exception as e:
            return self._response("error", "eda", errors=[f"EDA failed: {str(e)}"])
    
    def _smart_pipeline(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Smart end-to-end data pipeline based on natural language query"""
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = load_data(file_path)
            if df is None:
                return self._response("error", "pipeline", errors=["Could not load data"])
            
            original_shape = df.shape
            pipeline_steps = []
            
            # Step 1: Data profiling
            pipeline_steps.append("Data profiling completed")
            
            # Step 2: Automatic cleaning
            # Remove duplicates
            before_dup = len(df)
            df.drop_duplicates(inplace=True)
            if before_dup > len(df):
                pipeline_steps.append(f"Removed {before_dup - len(df)} duplicates")
            
            # Handle missing values
            missing_before = df.isnull().sum().sum()
            if missing_before > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                for col in numeric_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
                
                for col in categorical_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                
                pipeline_steps.append(f"Filled {missing_before} missing values")
            
            # Step 3: Feature engineering (if requested)
            if "feature" in query.lower() or "engineer" in query.lower():
                # Create datetime features
                datetime_cols = df.select_dtypes(include=['datetime64']).columns
                for col in datetime_cols:
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    pipeline_steps.append(f"Created date features from {col}")
            
            # Step 4: Encoding (if requested)
            if "encode" in query.lower() or "ml" in query.lower():
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if df[col].nunique() < 10:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        pipeline_steps.append(f"Encoded {col}")
            
            # Step 5: Export
            output_path = inputs.get("output_path", "data/processed_data.csv")
            df.to_csv(output_path, index=False)
            pipeline_steps.append(f"Exported to {output_path}")
            
            return self._response("success", "pipeline", {
                "message": f"Pipeline completed: {original_shape} → {df.shape}",
                "original_shape": original_shape,
                "final_shape": df.shape,
                "pipeline_steps": pipeline_steps,
                "output_path": output_path,
                "features_created": df.shape[1] - original_shape[1]
            })
            
        except Exception as e:
            return self._response("error", "pipeline", errors=[f"Pipeline failed: {str(e)}"])
    
    def _smart_general(self, query: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general data queries"""
        return self._response("success", "general", {
            "message": "I can help you with data cleaning, analysis, transformation, loading, exporting, EDA, and pipelines",
            "available_operations": [
                "Clean data (remove duplicates, handle missing values, outliers)",
                "Analyze data (statistics, correlations, insights)",
                "Transform data (scaling, encoding, feature engineering)",
                "Load data from various formats",
                "Export data to different formats",
                "Perform comprehensive EDA",
                "Run end-to-end data pipelines"
            ],
            "query_understood": query
        })
    
    def _extract_cleaning_operations(self, query: str) -> List[str]:
        """Extract specific cleaning operations from query"""
        operations = []
        
        if any(word in query for word in ["duplicate", "duplicates"]):
            operations.append("duplicates")
        if any(word in query for word in ["missing", "na", "null", "nan"]):
            operations.append("missing")
        if any(word in query for word in ["outlier", "outliers", "anomal"]):
            operations.append("outliers")
        
        return operations
    
    def _generate_data_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate insights from data analysis"""
        insights = []
        
        # Data size insights
        if len(df) > 100000:
            insights.append("Large dataset - consider sampling for faster processing")
        elif len(df) < 100:
            insights.append("Small dataset - results may not be statistically significant")
        
        # Missing data insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            insights.append(f"High missing data ({missing_pct:.1f}%) - requires attention")
        elif missing_pct == 0:
            insights.append("No missing values detected - data is complete")
        
        # Duplicate insights
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        if dup_pct > 5:
            insights.append(f"High duplicate rate ({dup_pct:.1f}%) - consider deduplication")
        
        # Column insights
        if len(df.columns) > 50:
            insights.append("High-dimensional dataset - consider feature selection")
        
        return insights
    
    def _generate_eda_insights(self, df: pd.DataFrame, eda_results: Dict) -> List[str]:
        """Generate insights from EDA results"""
        insights = []
        
        quality_score = eda_results.get("data_quality", {}).get("overall_quality_score", 0)
        if quality_score > 90:
            insights.append("Excellent data quality - ready for analysis")
        elif quality_score > 70:
            insights.append("Good data quality with minor issues")
        else:
            insights.append("Data quality needs improvement before analysis")
        
        # High correlation insights
        high_corr = eda_results.get("correlation_analysis", {}).get("high_correlations", [])
        if len(high_corr) > 0:
            insights.append(f"Found {len(high_corr)} highly correlated feature pairs")
        
        return insights
    
    def _generate_eda_recommendations(self, df: pd.DataFrame, eda_results: Dict) -> List[str]:
        """Generate recommendations from EDA results"""
        recommendations = []
        
        # Missing value recommendations
        missing_cols = eda_results.get("missing_analysis", {}).get("columns_with_missing", {})
        if len(missing_cols) > 0:
            recommendations.append("Handle missing values before modeling")
        
        # Correlation recommendations
        high_corr = eda_results.get("correlation_analysis", {}).get("high_correlations", [])
        if len(high_corr) > 0:
            recommendations.append("Consider removing highly correlated features")
        
        # Scaling recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            recommendations.append("Consider scaling numeric features for ML")
        
        return recommendations
    
    def _response(self, status: str, action: str, data: Optional[Dict] = None, errors: Optional[List] = None) -> Dict[str, Any]:
        """Generate standardized response"""
        response = {
            "status": status,
            "action": action,
            "agent": self.name
        }
        
        if data:
            response.update(data)
        
        if errors:
            response["errors"] = errors
            
        return response