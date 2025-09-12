"""  Visualiztion Agent for Visualize the  data and plots any report , graph etc"""
from app.mfrom app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from app.config import get_ollama_config
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import json
import base64
import re
from io import BytesIO

logger = get_logger(__name__)

class VisualizationAgent:
    """ Pure LLM Based visualization which generate any plot code dynamically"""
    def __init__(self):
        self.name =" VisualizationAgent"
        self.ollama_config = get_ollama_config()
        logger.info(f"Initialized {self.name} with pure LLM code generation")

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
            except Exception as e:
                return False

    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute visualization using pure LLM code generation"""
        try:
            query = task.inputs.get("query", "")
            data = task.inputs.get("data", [])
            statistical_results = task.inputs.get("statistical_analysis", {})
            
            # Use LLM to generate complete plot code
            return self._generate_plot_with_llm(query, data, statistical_results)
                
        except Exception as e:
            logger.error(f"VisualizationAgent execution failed: {str(e)}")
            return {"status": "error", "errors": [str(e)], "agent": self.name}
    
    def _generate_plot_with_llm(self, query: str, data: Any, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate complete plotting code with explanation and insights"""
        try:
            df = self._prepare_data(data)
            if df is None or df.empty:
                return {"status": "error", "errors": ["No valid data provided"]}
            
            # Get data info for LLM
            data_info = self._get_data_info(df)
            
            # Generate code AND explanation using enhanced LLM
            plot_code, explanation, insights = self._ask_llm_for_complete_analysis(query, data_info, stats)
            
            if not plot_code:
                return {"status": "error", "errors": ["Could not generate plot code"]}
            
            # Execute the generated code to create actual plot
            result = self._execute_generated_code(plot_code, df)
            
            # Generate data quality assessment
            data_quality = self._calculate_data_quality(df)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(df, stats, insights)
            
            return {
                "status": "success",
                "action": "enhanced_visualization_analysis",
                "output": {
                    "plot": result.get("plot_data", ""),     # Actual plot image
                    "code": plot_code,                        # Generated code
                    "explanation": explanation,               # Detailed explanation
                    "insights": insights,                     # AI-powered insights
                    "recommendations": recommendations,       # Action items
                    "format": result.get("format", "unknown")
                },
                "metadata": {
                    "data_shape": df.shape,
                    "statistical_context": stats,
                    "data_columns": list(df.columns),
                    "data_quality_score": data_quality,
                    "plot_complexity": self._assess_plot_complexity(plot_code),
                    "analysis_type": "comprehensive_enhanced"
                },
                "message": "Generated enhanced visualization with insights, code, and recommendations"
            }
            
        except Exception as e:
            return {"status": "error", "errors": [f"Enhanced plot generation failed: {str(e)}"]}
    def _ask_llm_for_complete_analysis(self, query: str, data_info: Dict, stats: Dict) -> tuple:
        """Ask LLM for plot code, explanation, and insights"""
        try:
            prompt = f"""
            Create a comprehensive visualization analysis for this request:
            
            User Query: "{query}"
            
            Data Information:
            - Shape: {data_info['shape']}
            - Columns: {data_info['columns']}
            - Numeric columns: {data_info['numeric_columns']}
            - Categorical columns: {data_info['categorical_columns']}
            - Sample data: {data_info.get('sample_data', {})}
            
            Statistical Analysis Results: {json.dumps(stats, indent=2)}
            
            Provide THREE parts in your response:
            
            PART 1 - Python Code:
            ```python
            # Complete plotting code here - make it publication-ready
            ```
            
            PART 2 - Explanation:
            Explain:
            1. Why this visualization answers the user's question
            2. What insights can be gained from this plot
            3. How to interpret the results
            4. What the code does step by step
            5. Benefits of this visualization approach
            
            PART 3 - Key Insights:
            Provide data insights in JSON format:
            {{
                "key_findings": ["Finding 1", "Finding 2"],
                "patterns": ["Pattern 1", "Pattern 2"],
                "anomalies": ["Anomaly 1"],
                "business_implications": ["Implication 1", "Implication 2"]
            }}
            
            Format your response exactly like this:
            
            CODE:
            ```python
            [your code here]
            ```
            
            EXPLANATION:
            [your detailed explanation here]
            
            INSIGHTS:
            [your JSON insights here]
            """
            
            import requests
            response = requests.post(
                f"{self.ollama_config['base_url']}/api/generate",
                json={
                    "model": self.ollama_config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Extract code
                code_match = re.search(r'```python\n(.*?)\n```', llm_output, re.DOTALL)
                code = code_match.group(1).strip() if code_match else ""
                
                # Extract explanation
                explanation_match = re.search(r'EXPLANATION:\s*(.*?)(?=INSIGHTS:|$)', llm_output, re.DOTALL)
                explanation = explanation_match.group(1).strip() if explanation_match else "Visualization generated successfully."
                
                # Extract insights
                insights_match = re.search(r'INSIGHTS:\s*(\{.*?\})', llm_output, re.DOTALL)
                if insights_match:
                    try:
                        insights = json.loads(insights_match.group(1))
                    except:
                        insights = {"key_findings": ["Analysis completed"], "patterns": [], "anomalies": [], "business_implications": []}
                else:
                    insights = {"key_findings": ["Visualization created"], "patterns": [], "anomalies": [], "business_implications": []}
                
                return code, explanation, insights
            
            return "", "Could not generate analysis.", {}
            
        except Exception as e:
            logger.error(f"LLM analysis generation failed: {e}")
            return "", f"Error generating analysis: {str(e)}", {}

    def _execute_generated_code(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute LLM-generated plotting code"""
        try:
            # Auto-install common packages
            packages = ['matplotlib', 'seaborn', 'scipy', 'plotly']
            for pkg in packages:
                self._auto_install(pkg)
            
            # Set up execution environment
            exec_globals = {
                'df': df,
                'np': np,
                'pd': pd,
                'json': json
            }
            
            # Import common plotting libraries
            try:
                import matplotlib.pyplot as plt
                exec_globals['plt'] = plt
            except ImportError:
                pass
            
            try:
                import seaborn as sns
                exec_globals['sns'] = sns
            except ImportError:
                pass
            
            try:
                from scipy import stats
                exec_globals['stats'] = stats
            except ImportError:
                pass
            
            try:
                import plotly.express as px
                import plotly.graph_objects as go
                exec_globals['px'] = px
                exec_globals['go'] = go
            except ImportError:
                pass
            
            # Execute the generated code
            exec(code, exec_globals)
            
            # Try to save the plot
            if 'plt' in exec_globals:
                # Matplotlib-based plot
                buffer = BytesIO()
                plt = exec_globals['plt']
                plt.tight_layout()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                return {"plot_data": plot_data, "format": "base64_png"}
            
            elif 'fig' in exec_globals:
                # Plotly figure
                fig = exec_globals['fig']
                plot_html = fig.to_html()
                return {"plot_data": plot_html, "format": "html"}
            
            else:
                return {"plot_data": "Plot generated but format unknown", "format": "unknown"}
                
        except Exception as e:
            return {"plot_data": f"Code execution failed: {str(e)}", "format": "error"}

    def _get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data information for LLM"""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns}
        }
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and provide score"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            missing_cells = df.isnull().sum().sum()
            
            quality_metrics = {
                "completeness": (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 100,
                "uniqueness": df.nunique().mean() / len(df) * 100 if len(df) > 0 else 0,
                "consistency": 85.0,  # Simplified metric
                "validity": 90.0,     # Simplified metric
            }
            
            # Calculate overall score
            quality_metrics["overall_score"] = (
                quality_metrics["completeness"] * 0.4 +
                quality_metrics["uniqueness"] * 0.2 +
                quality_metrics["consistency"] * 0.2 +
                quality_metrics["validity"] * 0.2
            )
            
            return quality_metrics
        except Exception as e:
            return {"error": f"Quality assessment failed: {str(e)}"}
    
    def _assess_plot_complexity(self, code: str) -> Dict[str, Any]:
        """Assess the complexity of generated plot"""
        try:
            complexity_indicators = {
                "lines_of_code": len(code.split('\n')),
                "libraries_used": len([lib for lib in ['matplotlib', 'seaborn', 'plotly', 'scipy'] if lib in code]),
                "plot_elements": code.count('plt.') + code.count('sns.') + code.count('fig.'),
                "statistical_elements": code.count('stats.') + code.count('correlation') + code.count('regression'),
                "customization_level": code.count('color') + code.count('style') + code.count('theme')
            }
            
            # Calculate complexity score
            total_score = sum(complexity_indicators.values())
            complexity_level = "Simple" if total_score < 10 else "Moderate" if total_score < 20 else "Complex"
            
            return {
                **complexity_indicators,
                "complexity_level": complexity_level,
                "complexity_score": total_score
            }
        except Exception as e:
            return {"error": f"Complexity assessment failed: {str(e)}"}
    
    def _generate_recommendations(self, df: pd.DataFrame, stats: Dict, insights: Dict) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis"""
        try:
            recommendations = {
                "immediate_actions": [],
                "further_analysis": [],
                "data_improvements": [],
                "visualization_alternatives": []
            }
            
            # Data quality recommendations
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            if missing_pct > 5:
                recommendations["data_improvements"].append(f"Address {missing_pct:.1f}% missing data")
            
            # Statistical recommendations
            if 'shapiro_wilk' in stats and not stats['shapiro_wilk'].get('is_normal', True):
                recommendations["further_analysis"].append("Consider data transformation for normality")
            
            if 'outlier_analysis' in stats and stats['outlier_analysis'].get('outlier_percentage', 0) > 5:
                recommendations["immediate_actions"].append("Investigate outliers - high percentage detected")
            
            # Default recommendations if none generated
            if not any(recommendations.values()):
                recommendations["immediate_actions"] = ["Review visualization for insights"]
                recommendations["further_analysis"] = ["Consider additional statistical analysis"]
            
            return recommendations
        except Exception as e:
            return {"error": f"Recommendation generation failed: {str(e)}"}
    
    def _prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Convert any data format to DataFrame"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if isinstance(data, list):
                if len(data) == 0:
                    return None
                if isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame({"values": data})
            
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            
            elif isinstance(data, pd.DataFrame):
                return data
            
            return None
                
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return Noneory']).columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: int(df[col].nunique()) for col in df.columns}
        

    def _prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Convert any data format to DataFrame"""
        try:
            if isinstance(data, str):
                data = json.loads(data)
            
            if isinstance(data, list):
                if len(data) == 0:
                    return None
                if isinstance(data[0], dict):
                    return pd.DataFrame(data)
                else:
                    return pd.DataFrame({"values": data})
            
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            
            elif isinstance(data, pd.DataFrame):
                return data
            
            return None
                
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return None


