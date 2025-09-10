"""Agent for data wrangling and processing. Or Making all data Processing Task"""
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = get_logger(__name__)

class DataAgent:
    """Agent for data wrangling and processing tasks."""
    def __init__(self):
        self.name = "DataAgent"
        self.tools = ["pandas", "numpy", "polars", "dask", "pyjanitor", "sklearn"]
        self.allowed_dirs = ["data", "datasets", "uploads"]  # Security: restrict file access
        self.max_file_size = 100 * 1024 * 1024  # 100MB limit
        logger.info(f"Initialized {self.name} with tools: {self.tools}")

    def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security."""
        if not file_path:
            logger.warning("Empty file path provided")
            return False
        
        try:
            path = Path(file_path).resolve()
            # Check if file is in allowed directories
            is_allowed = any(allowed_dir in str(path) for allowed_dir in self.allowed_dirs)
            if not is_allowed:
                logger.warning(f"File path not in allowed directories: {file_path}")
            return is_allowed
        except (OSError, ValueError) as e:
            logger.error(f"Path validation failed for {file_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating path {file_path}: {str(e)}")
            return False

    def _load_dataframe(self, file_path: str) -> pd.DataFrame:
        """Securely load DataFrame from various formats."""
        if not self._validate_file_path(file_path):
            raise ValueError(f"File path not allowed: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.stat().st_size > self.max_file_size:
            raise ValueError(f"File too large: {path.stat().st_size} bytes")
        
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        elif path.suffix.lower() == ".json":
            return pd.read_json(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute the data wrangling task."""
        logger.info(f"{self.name} executing task: {task.task_id}")
        
        try:
            query = task.inputs.get("query", "")
            if "eda" in query.lower() or "comprehensive" in query.lower():
                return self.comprehensive_eda(task.inputs)
            elif "pipeline" in query.lower() or "full" in query.lower():
                return self.run_full_pipeline(task.inputs)
            elif "clean" in query.lower() or "prepare" in query.lower():
                return self.clean_data(task.inputs)
            elif "analyze" in query.lower() or "profile" in query.lower():
                return self.analyze_data(task.inputs)
            elif "load" in query.lower() or "read" in query.lower():
                return self.load_data(task.inputs)
            elif "transform" in query.lower() or "feature" in query.lower():
                return self.transform_data(task.inputs)
            elif "export" in query.lower() or "save" in query.lower():
                return self.export_data(task.inputs)
            else:
                return self.process_data(task.inputs)
        
        except Exception as e:
            logger.error(f"{self.name} execution failed: {str(e)}")
            return {"status": "error", "action": "execute", "errors": [str(e)]}

    def comprehensive_eda(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Complete EDA with statistical analysis."""
        logger.info("Running comprehensive EDA...")
        
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = self._load_dataframe(file_path)
            
            # Generate data profile
            profile = self.generate_data_profile(df)
            
            # Correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            correlation_matrix = {}
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr().to_dict()
            
            # Distribution analysis
            distribution_analysis = {}
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                distribution_analysis[col] = {
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                    "normality_test": stats.normaltest(df[col].dropna())[1] > 0.05
                }
            
            # Missing value patterns
            missing_patterns = {
                "columns_with_missing": df.columns[df.isnull().any()].tolist(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
            }
            
            # Data quality score
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
            quality_score = (completeness + uniqueness) / 2
            
            return {
                "status": "success",
                "action": "comprehensive_eda",
                "data_profile": profile,
                "correlation_analysis": correlation_matrix,
                "distribution_analysis": distribution_analysis,
                "missing_patterns": missing_patterns,
                "data_quality_score": round(quality_score, 2),
                "insights": self._generate_insights(df, profile),
                "recommendations": self._generate_recommendations(df, profile)
            }
        except Exception as e:
            return {"status": "error", "action": "comprehensive_eda", "errors": [f"EDA failed: {str(e)}"]}
    
    def generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        profile = {
            "shape": df.shape,
            "column_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "cardinality_analysis": {},
            "anomaly_detection": {}
        }
        
        # Cardinality analysis
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            unique_count = df[col].nunique()
            profile["cardinality_analysis"][col] = {
                "unique_values": unique_count,
                "cardinality_level": "high" if unique_count > len(df) * 0.5 else "medium" if unique_count > 10 else "low"
            }
        
        # Anomaly detection for numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            profile["anomaly_detection"][col] = {
                "outlier_count": len(outliers),
                "outlier_percentage": round(len(outliers) / len(df) * 100, 2)
            }
        
        return profile
    
    def create_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Automated feature creation."""
        df_features = df.copy()
        feature_types = config.get("feature_types", [])
        
        # Date/time features
        if "datetime" in feature_types:
            datetime_cols = df_features.select_dtypes(include=['datetime64']).columns
            for col in datetime_cols:
                df_features[f"{col}_year"] = df_features[col].dt.year
                df_features[f"{col}_month"] = df_features[col].dt.month
                df_features[f"{col}_day"] = df_features[col].dt.day
                df_features[f"{col}_weekday"] = df_features[col].dt.weekday
                df_features[f"{col}_quarter"] = df_features[col].dt.quarter
        
        # Mathematical features
        if "mathematical" in feature_types:
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                if df_features[col].min() > 0:  # Only for positive values
                    df_features[f"{col}_log"] = np.log1p(df_features[col])
                    df_features[f"{col}_sqrt"] = np.sqrt(df_features[col])
        
        # Interaction features
        if "interactions" in feature_types:
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                df_features[f"{col1}_{col2}_product"] = df_features[col1] * df_features[col2]
                df_features[f"{col1}_{col2}_ratio"] = df_features[col1] / (df_features[col2] + 1e-8)
        
        return df_features
    
    def advanced_encoding(self, df: pd.DataFrame, encoding_config: Dict[str, Any]) -> pd.DataFrame:
        """Smart encoding strategies."""
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()
            
            if unique_count <= 5:  # Low cardinality - OneHot
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded = encoder.fit_transform(df_encoded[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df_encoded.index)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), encoded_df], axis=1)
            
            elif unique_count <= 20:  # Medium cardinality - Label Encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
            
            else:  # High cardinality - Keep top categories, others as 'Other'
                top_categories = df_encoded[col].value_counts().head(10).index
                df_encoded[col] = df_encoded[col].apply(lambda x: x if x in top_categories else 'Other')
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
        
        return df_encoded
    
    def run_full_pipeline(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute end-to-end data pipeline."""
        logger.info("Running full data pipeline...")
        
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = self._load_dataframe(file_path)
            pipeline_results = {"steps_completed": []}
            
            # Step 1: Data Profiling
            profile = self.generate_data_profile(df)
            pipeline_results["steps_completed"].append("data_profiling")
            
            # Step 2: Auto-recommend and apply cleaning
            recommendations = self.auto_recommend_pipeline(df)
            
            # Apply cleaning based on recommendations
            if recommendations["needs_cleaning"]:
                df.drop_duplicates(inplace=True)
                df.fillna(df.mean(numeric_only=True), inplace=True)
                pipeline_results["steps_completed"].append("data_cleaning")
            
            # Step 3: Feature Engineering
            if inputs.get("create_features", True):
                feature_config = {"feature_types": ["datetime", "mathematical", "interactions"]}
                df = self.create_features(df, feature_config)
                pipeline_results["steps_completed"].append("feature_creation")
            
            # Step 4: Encoding
            if inputs.get("apply_encoding", True):
                df = self.advanced_encoding(df, {})
                pipeline_results["steps_completed"].append("encoding")
            
            # Step 5: Export
            output_path = inputs.get("output_path", "data/processed_data.csv")
            df.to_csv(output_path, index=False)
            pipeline_results["steps_completed"].append("export")
            
            return {
                "status": "success",
                "action": "full_pipeline",
                "original_shape": profile["shape"],
                "final_shape": df.shape,
                "features_created": df.shape[1] - profile["shape"][1],
                "pipeline_results": pipeline_results,
                "output_path": output_path,
                "recommendations_applied": recommendations,
                "summary": f"Pipeline completed: {profile['shape']} → {df.shape}, {len(pipeline_results['steps_completed'])} steps"
            }
        except Exception as e:
            return {"status": "error", "action": "full_pipeline", "errors": [f"Pipeline failed: {str(e)}"]}
    
    def auto_recommend_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """AI-powered pipeline recommendations."""
        recommendations = {
            "needs_cleaning": False,
            "remove_outliers": False,
            "encoding_strategy": "auto",
            "feature_engineering": [],
            "scaling_needed": False
        }
        
        # Check if cleaning is needed
        if df.isnull().sum().sum() > 0 or df.duplicated().sum() > 0:
            recommendations["needs_cleaning"] = True
        
        # Check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_percentage = 0
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_percentage += len(outliers) / len(df)
        
        if outlier_percentage > 0.05:  # More than 5% outliers
            recommendations["remove_outliers"] = True
        
        # Feature engineering recommendations
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            recommendations["feature_engineering"].append("datetime_features")
        
        if len(numeric_cols) > 1:
            recommendations["feature_engineering"].append("interaction_features")
            recommendations["scaling_needed"] = True
        
        return recommendations
    
    def _generate_insights(self, df: pd.DataFrame, profile: Dict[str, Any]) -> List[str]:
        """Generate automated insights."""
        insights = []
        
        # Data size insights
        if df.shape[0] > 100000:
            insights.append("Large dataset detected - consider sampling for faster processing")
        
        # Missing data insights
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        if missing_percentage > 20:
            insights.append(f"High missing data ({missing_percentage:.1f}%) - requires attention")
        
        # Cardinality insights
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() > len(df) * 0.8:
                insights.append(f"Column '{col}' has very high cardinality - consider grouping")
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame, profile: Dict[str, Any]) -> List[str]:
        """Generate automated recommendations."""
        recommendations = []
        
        # Cleaning recommendations
        if df.duplicated().sum() > 0:
            recommendations.append("Remove duplicate rows")
        
        if df.isnull().sum().sum() > 0:
            recommendations.append("Handle missing values with appropriate imputation")
        
        # Feature engineering recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            recommendations.append("Consider feature scaling for machine learning")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            recommendations.append("Apply appropriate encoding for categorical variables")
        
        return recommendations

    def load_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load data from a specified source. like csv, json etc"""
        logger.info("Loading data...")

        data_source = inputs.get("data_source") or inputs.get("file_path")
        if not data_source:
            return {"status": "error", "action": "load_data", "errors": ["No data source provided"]}
        
        try:
            df = self._load_dataframe(data_source)
            
            # Basic data quality checks
            null_counts = df.isnull().sum()
            duplicate_count = df.duplicated().sum()
            
            return {
                "status": "success",
                "action": "data_loading",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "null_counts": null_counts.to_dict(),
                "duplicate_rows": int(duplicate_count),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "summary": f"Loaded {len(df)} rows × {len(df.columns)} columns",
            }
        except Exception as e:
            return {"status": "error", "action": "data_loading", "errors": [f"Failed to load data: {str(e)}"]}

    def clean_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and preprocess data (duplicates, missing values)."""
        logger.info("Cleaning data...")

        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = self._load_dataframe(file_path)
            original_rows = len(df)
            
            cleaning_strategy = inputs.get("strategy", "basic")
            
            # Remove duplicates
            df.drop_duplicates(inplace=True)
            duplicates_removed = original_rows - len(df)

            # Advanced missing value handling
            missing_before = int(df.isnull().sum().sum())
            
            if cleaning_strategy == "advanced":
                # Use sklearn imputers for better handling
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                
                if len(numeric_cols) > 0:
                    imputer_num = SimpleImputer(strategy='median')
                    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
                
                if len(categorical_cols) > 0:
                    imputer_cat = SimpleImputer(strategy='most_frequent')
                    df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
            else:
                # Basic strategy
                for col in df.columns:
                    if df[col].isnull().any():
                        if df[col].dtype in [np.float64, np.int64]:
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            mode_values = df[col].mode()
                            if len(mode_values) > 0:
                                df[col].fillna(mode_values[0], inplace=True)
                            else:
                                df[col].fillna('Unknown', inplace=True)

            missing_after = int(df.isnull().sum().sum())
            missing_filled = missing_before - missing_after
            
            # Outlier detection and handling
            outliers_removed = 0
            if inputs.get("remove_outliers", False):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outliers_removed += len(outliers)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            cleaned_rows = len(df)
            
            # Save cleaned data if output path provided
            output_path = inputs.get("output_path")
            if output_path:
                df.to_csv(output_path, index=False)

            return {
                "status": "success",
                "action": "data_cleaning",
                "original_rows": original_rows,
                "cleaned_rows": cleaned_rows,
                "duplicates_removed": duplicates_removed,
                "missing_values_filled": missing_filled,
                "outliers_removed": outliers_removed,
                "columns": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "cleaned_data": df.to_dict('records') if inputs.get("return_data", False) else None,
                "output_saved": output_path if output_path else None,
                "summary": f"Cleaned dataset → {cleaned_rows} rows, "
                           f"{duplicates_removed} duplicates removed, "
                           f"{missing_filled} missing filled, "
                           f"{outliers_removed} outliers removed",
            }
        except Exception as e:
            return {"status": "error", "action": "data_cleaning", "errors": [f"Cleaning failed: {str(e)}"]}

    def analyze_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Profile dataset for insights and statistics."""
        logger.info("Analyzing data...")

        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = self._load_dataframe(file_path)
            rows, cols = df.shape

            # Basic statistics
            stats = {
                "total_records": rows,
                "total_features": cols,
                "missing_values": int(df.isnull().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "memory_usage_MB": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            }

            # Column analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            column_analysis = {
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "datetime_columns": len(datetime_cols),
                "numeric_column_names": numeric_cols,
                "categorical_column_names": categorical_cols,
            }

            # Statistical summary for numeric columns
            numeric_summary = {}
            if len(numeric_cols) > 0:
                numeric_summary = df[numeric_cols].describe().to_dict()

            # Categorical analysis
            categorical_summary = {}
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                categorical_summary[col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }

            insights = [
                f"Dataset has {rows} rows and {cols} columns",
                f"Detected {stats['missing_values']} missing values ({stats['missing_values']/(rows*cols)*100:.1f}%)",
                f"Found {stats['duplicate_rows']} duplicate rows ({stats['duplicate_rows']/rows*100:.1f}%)",
                f"Contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features",
            ]

            recommendations = []
            if stats['missing_values'] > 0:
                recommendations.append("Handle missing values before analysis")
            if stats['duplicate_rows'] > 0:
                recommendations.append("Remove duplicate rows")
            if len(numeric_cols) > 0:
                recommendations.append("Consider scaling numerical features")
            if len(categorical_cols) > 0:
                recommendations.append("Encode categorical variables for ML")

            return {
                "status": "success",
                "action": "data_analysis",
                "statistics": stats,
                "column_analysis": column_analysis,
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary,
                "insights": insights,
                "recommendations": recommendations,
            }
        except Exception as e:
            return {"status": "error", "action": "data_analysis", "errors": [f"Analysis failed: {str(e)}"]}

    def transform_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data with feature engineering."""
        logger.info("Transforming data...")
        
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            df = self._load_dataframe(file_path)
            
            transformations = inputs.get("transformations", [])
            results = {"applied_transformations": []}
            
            for transform in transformations:
                if transform == "scale_numeric":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        scaler = StandardScaler()
                        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                        results["applied_transformations"].append("StandardScaler applied to numeric columns")
                
                elif transform == "encode_categorical":
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        if df[col].nunique() < 10:  # Only encode low-cardinality columns
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            results["applied_transformations"].append(f"LabelEncoder applied to {col}")
            
            return {
                "status": "success",
                "action": "data_transformation",
                "rows": len(df),
                "columns": len(df.columns),
                "transformations_applied": len(results["applied_transformations"]),
                "details": results["applied_transformations"],
            }
        except Exception as e:
            return {"status": "error", "action": "data_transformation", "errors": [f"Transformation failed: {str(e)}"]}
    
    def export_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Export processed data to various formats."""
        logger.info("Exporting data...")
        
        try:
            file_path = inputs.get("data_source") or inputs.get("file_path")
            output_path = inputs.get("output_path")
            output_format = inputs.get("format", "csv")
            
            if not output_path:
                return {"status": "error", "action": "data_export", "errors": ["No output path provided"]}
            
            df = self._load_dataframe(file_path)
            
            if output_format.lower() == "csv":
                df.to_csv(output_path, index=False)
            elif output_format.lower() == "json":
                df.to_json(output_path, orient='records')
            elif output_format.lower() == "parquet":
                df.to_parquet(output_path)
            else:
                return {"status": "error", "action": "data_export", "errors": [f"Unsupported format: {output_format}"]}
            
            return {
                "status": "success",
                "action": "data_export",
                "output_path": output_path,
                "format": output_format,
                "rows_exported": len(df),
                "file_size_mb": round(Path(output_path).stat().st_size / 1024 / 1024, 2),
            }
        except Exception as e:
            return {"status": "error", "action": "data_export", "errors": [f"Export failed: {str(e)}"]}

    def process_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback generic processor."""
        logger.info("Generic data processing...")
        return {
            "status": "success",
            "action": "data_processing",
            "message": "No specific action matched. Available actions: load, clean, analyze, transform, export, eda, pipeline",
            "available_actions": ["load", "clean", "analyze", "transform", "export", "eda", "pipeline"],
        }