#  This agent can be check the output which given from all agent and also check the quality of output
from app.mcp.mcp_schema import TaskMessage
from app.utils.logger import get_logger, log_execution_time
from typing import Dict, Any, List, Optional, Set
import ast
import re
import json
import subprocess
import tempfile
from pathlib import Path
try:
    from app.config import OLLAMA_CONFIG
except ImportError:
    OLLAMA_CONFIG = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "timeout": 30
    }
logger = get_logger(__name__)
class CriticAgent:
    """Agent for quality review and improvement recommendations."""
    def __init__(self):
        self.name ="CriticAgent"
        self.available_tools = self._check_available_tools()
        self.thresholds = {
            "accuracy": 0.7,
            "overfitting_gap": 0.1,
            "complexity": 20,
            "precision": 0.6,
            "recall": 0.6,
            "f1_score": 0.6
        }
        logger.info(f"Initialized {self.name} with tools :{list(self.available_tools.keys())}")
        
    def _check_available_tools(self) -> Dict[str ,bool] :
        """ Check which external tools are available"""
        tools = {}
        for tool in ["flake8", "bandit", "radon", "pylint"]:
            try:
                subprocess.run([tool, "--version"], capture_output=True, timeout=5)
                tools[tool] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tools[tool] = False
                logger.warning(f"Tool {tool} not available")
        return tools
    
    @log_execution_time
    def execute(self, task: TaskMessage) -> Dict[str, Any]:
        """Execute the quality review task."""
        logger.info(f"{self.name} executing task: {task.task_id}")
        try :
            #update the thresholds form inputs
            self.thresholds.update(task.inputs.get("thresholds",{}))
            query = task.inputs.get("query", "")
            if "code" in query.lower():
                return self.review_code(task.inputs)
            elif "model" in query.lower():
                return self.review_model(task.inputs)
            elif "output" in query.lower():
                return self.review_agent_output(task.inputs)
            
            else :
                return self.comprehensive_review(task.inputs)
        
        except Exception as e:
            logger.error(f"{self.name} execution failed: {str(e)}")
            return self._standard_response("error", "execute", errors=[str(e)])

    def review_code(self , inputs : Dict[str , Any]) -> Dict[str , Any]:
        """ Review the code quality using tools"""
        logger.info("Reviewing the code...")
        try :
            code_content = inputs.get("code_content", "")
            code_path = inputs.get("code_path")

            # check the code_content and code path is  available
            if  not code_content  and not code_path :
                return self._standard_response("error", "code_review", errors=["No code content or path provided"])
            
            issues = []
            tool_failures = []
            # create the temp file if only the content is provided
            temp_file = None
            if code_content and not code_path :
                temp_file = tempfile.NamedTemporaryFile(mode ='w', suffix ='.py' , delete = False)
                temp_file.write(code_content)
                temp_file.close()
                code_path = temp_file.name
            
            try:
                # Run available static analysis tools
                linter_issues, linter_failures = self._run_linters(code_path)
                issues.extend(linter_issues)
                tool_failures.extend(linter_failures)
                
                # AST analysis (always available)
                if not code_content:
                    with open(code_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                
                ast_issues = self._analyze_ast(code_content)
                issues.extend(ast_issues)
                
                # Deduplicate issues
                issues = self._deduplicate_issues(issues)
                
                # Calculate complexity
                complexity = self._calculate_complexity(code_content, code_path)
                
                # Enhanced analysis with new features
                doc_issues = self._analyze_documentation(code_content)
                issues.extend(doc_issues)
                
                code_smells = self._detect_code_smells(code_content)
                issues.extend(code_smells)
                
                # AI-powered review
                ai_review = self._ai_code_review(code_content, inputs.get("agent_name", "unknown"))
                
                # Generate code fixes
                code_fixes = self._generate_code_fixes(issues)
                
                # Final deduplication
                issues = self._deduplicate_issues(issues)
                
                return self._standard_response("success", "code_review", {
                    "issues": issues,
                    "complexity_score": complexity,
                    "tool_failures": tool_failures,
                    "tools_missing": [tool for tool, available in self.available_tools.items() if not available],
                    "recommendations": self._generate_code_recommendations(issues, complexity),
                    "code_fixes": code_fixes,
                    "ai_review": ai_review,
                    "total_issues": len(issues),
                    "quality_score": max(0, 100 - len(issues) * 5 - max(0, complexity - 10) * 2)
                })
            
            finally:
                # Cleanup temp file
                if temp_file:
                    Path(temp_file.name).unlink(missing_ok=True)
        
        except Exception as e:
            return self._standard_response("error", "code_review", errors=[f"Code review failed: {str(e)}"])
    
    def review_model(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Review ML model Performance"""
        logger.info("Reviewing the model...")

        try:
            metrics = inputs.get("model_metrics", {})
            train_metrics = inputs.get("train_metrics", {})
            val_metrics = inputs.get("val_metrics", {})

            issues = []

            # Performance analysis with configurable threshold
            issues.extend(self._analyze_performance_metrics(metrics, val_metrics))

            # Overfitting detection
            if train_metrics and val_metrics:
                issues.extend(self._detect_overfitting(train_metrics, val_metrics))
            
            model_score = self._calculate_model_score(issues)
            return self._standard_response("success", "model_review", {
                "issues": issues,
                "model_score": model_score,
                "thresholds_used": self.thresholds,
                "recommendations": self._generate_model_recommendations(issues)
            })
        
        except Exception as e:
            return self._standard_response("error", "model_review", errors=[f"Model review failed: {str(e)}"])
    
    def review_agent_output(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Review the agent output"""
        logger.info("Reviewing The Agent Output")

        try:
            agent_output = inputs.get("agent_output", {})
            agent_name = inputs.get("agent_name", "unknown")
            issues = self._validate_output_structure(agent_output)
            quality_score = max(0, 100 - len(issues) * 10)
            
            return self._standard_response("success", "output_review", {
                "agent_name": agent_name,
                "issues": issues,
                "quality_score": quality_score,
                "recommendations": self._generate_output_recommendations(issues)
            })
        
        except Exception as e:
            return self._standard_response("error", "output_review", errors=[f"Output review failed: {str(e)}"])

    
    def comprehensive_review(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality review."""
        logger.info("Performing comprehensive review...")
        
        try:
            results = {}
            
            if inputs.get("code_content") or inputs.get("code_path"):
                results["code_review"] = self.review_code(inputs)
            
            if inputs.get("model_metrics"):
                results["model_review"] = self.review_model(inputs)
            
            if inputs.get("agent_output"):
                results["output_review"] = self.review_agent_output(inputs)
            
            overall_score = self._calculate_overall_score(results)
            
            return self._standard_response("success", "comprehensive_review", {
                "results": results,
                "overall_score": overall_score,
                "grade": self._assign_grade(overall_score)
            })
        
        except Exception as e:
            return self._standard_response("error", "comprehensive_review", errors=[f"Review failed: {str(e)}"])

    def _run_linters(self, code_path: str) -> tuple[List[Dict], List[str]]:
        """Run available linters safely."""
        issues = []
        failures = []
        
        # Flake8
        if self.available_tools.get("flake8"):
            try:
                result = subprocess.run(
                    ["flake8", "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s", code_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if ':' in line:
                            parts = line.split(':', 3)
                            if len(parts) >= 4:
                                issues.append({
                                    "tool": "flake8",
                                    "type": "style_issue",
                                    "line": int(parts[1]) if parts[1].isdigit() else 0,
                                    "message": parts[3].strip(),
                                    "severity": "low",
                                    "location": f"{parts[1]}:{parts[2]}"
                                })
            except Exception as e:
                failures.append(f"flake8: {str(e)}")
        
        # Bandit
        if self.available_tools.get("bandit"):
            try:
                result = subprocess.run(
                    ["bandit", "-f", "json", code_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    for issue in bandit_data.get("results", []):
                        issues.append({
                            "tool": "bandit",
                            "type": "security_issue",
                            "line": issue.get("line_number", 0),
                            "message": issue.get("issue_text", ""),
                            "severity": issue.get("issue_severity", "medium").lower(),
                            "location": f"{issue.get('line_number', 0)}"
                        })
            except Exception as e:
                failures.append(f"bandit: {str(e)}")
        
        return issues, failures

    def _calculate_complexity(self, code_content: str, code_path: Optional[str]) -> int:
        """Calculate complexity using radon or fallback."""
        if self.available_tools.get("radon") and code_path:
            try:
                result = subprocess.run(
                    ["radon", "cc", "-s", code_path],
                    capture_output=True, text=True, timeout=30
                )
                if result.stdout:
                    # Parse radon output for average complexity
                    lines = result.stdout.strip().split('\n')
                    complexities = []
                    for line in lines:
                        if '(' in line and ')' in line:
                            match = re.search(r'\((\d+)\)', line)
                            if match:
                                complexities.append(int(match.group(1)))
                    return sum(complexities) if complexities else 1
            except Exception:
                pass
        
        # Fallback: count decision points
        decision_keywords = ["if", "elif", "while", "for", "try", "except", "and", "or"]
        return sum(len(re.findall(rf"\b{kw}\b", code_content)) for kw in decision_keywords) + 1

    def _analyze_ast(self, code_content: str) -> List[Dict[str, Any]]:
        """Analyze code using AST."""
        issues = []
        
        try:
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                # Dangerous function calls
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec"]:
                        issues.append({
                            "type": "dangerous_function",
                            "function": node.func.id,
                            "line": node.lineno,
                            "severity": "critical",
                            "location": str(node.lineno)
                        })
                
                # Complex functions
                if isinstance(node, ast.FunctionDef) and len(node.body) > 25:
                    issues.append({
                        "type": "complex_function",
                        "function": node.name,
                        "lines": len(node.body),
                        "line": node.lineno,
                        "severity": "medium",
                        "location": str(node.lineno)
                    })
        
        except SyntaxError as e:
            issues.append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno or 0,
                "severity": "high",
                "location": str(e.lineno or 0)
            })
        
        return issues

    def _deduplicate_issues(self, issues: List[Dict]) -> List[Dict]:
        """Remove duplicate issues based on location and type."""
        seen: Set[str] = set()
        unique_issues = []
        
        for issue in issues:
            key = f"{issue.get('type')}:{issue.get('location', '')}"
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        return unique_issues

    def _analyze_performance_metrics(self, metrics: Dict, val_metrics: Dict) -> List[Dict]:
        """Analyze performance metrics with configurable thresholds."""
        issues = []
        
        # Use validation metrics if available, otherwise general metrics
        active_metrics = val_metrics if val_metrics else metrics
        
        for metric_name, threshold in self.thresholds.items():
            if metric_name in active_metrics:
                value = active_metrics[metric_name]
                if value < threshold:
                    issues.append({
                        "type": f"low_{metric_name}",
                        "value": value,
                        "threshold": threshold,
                        "severity": "high" if metric_name == "accuracy" else "medium"
                    })
        
        return issues

    def _detect_overfitting(self, train_metrics: Dict, val_metrics: Dict) -> List[Dict]:
        """Detect overfitting with configurable gap threshold."""
        issues = []
        
        for metric in ["accuracy", "f1_score"]:
            train_val = train_metrics.get(metric)
            val_val = val_metrics.get(metric)
            
            if train_val and val_val:
                gap = train_val - val_val
                if gap > self.thresholds["overfitting_gap"]:
                    issues.append({
                        "type": "overfitting",
                        "metric": metric,
                        "train_value": train_val,
                        "val_value": val_val,
                        "gap": gap,
                        "threshold": self.thresholds["overfitting_gap"],
                        "severity": "high"
                    })
        
        return issues

    def _validate_output_structure(self, output: Any) -> List[Dict]:
        """Validate agent output structure."""
        issues = []
        
        if not isinstance(output, dict):
            issues.append({
                "type": "invalid_structure",
                "message": "Output must be a dictionary",
                "severity": "high"
            })
            return issues
        
        # Required fields
        required = ["status", "action"]
        for field in required:
            if field not in output:
                issues.append({
                    "type": "missing_field",
                    "field": field,
                    "severity": "medium"
                })
        
        # Error handling validation
        if output.get("status") == "error" and "errors" not in output:
            issues.append({
                "type": "poor_error_handling",
                "message": "Error status without error details",
                "severity": "medium"
            })
        
        return issues

    def _calculate_model_score(self, issues: List[Dict]) -> int:
        """Calculate model quality score."""
        score = 100
        weights = {"critical": 25, "high": 15, "medium": 10, "low": 5}
        
        for issue in issues:
            weight = weights.get(issue.get("severity", "low"), 5)
            score -= weight
        
        return max(0, score)

    def _calculate_overall_score(self, results: Dict) -> int:
        """Calculate overall quality score."""
        scores = []
        
        for result in results.values():
            if result.get("status") == "success":
                if "quality_score" in result:
                    scores.append(result["quality_score"])
                elif "model_score" in result:
                    scores.append(result["model_score"])
        
        return sum(scores) // len(scores) if scores else 0

    def _assign_grade(self, score: int) -> str:
        """Assign letter grade based on score."""
        if score >= 90: return "A"
        elif score >= 80: return "B"
        elif score >= 70: return "C"
        elif score >= 60: return "D"
        else: return "F"

    def _generate_code_recommendations(self, issues: List, complexity: int) -> List[str]:
        """Generate actionable code recommendations."""
        recommendations = []
        
        if complexity > self.thresholds["complexity"]:
            recommendations.append(f"Reduce complexity (current: {complexity}, threshold: {self.thresholds['complexity']})")
        
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        if critical_issues:
            recommendations.append("Address critical security issues immediately")
        
        return recommendations

    def _generate_model_recommendations(self, issues: List) -> List[str]:
        """Generate model improvement recommendations."""
        recommendations = []
        
        if any(i.get("type") == "overfitting" for i in issues):
            recommendations.append("Add regularization or increase validation data")
        
        low_metrics = [i for i in issues if i.get("type", "").startswith("low_")]
        if low_metrics:
            recommendations.append("Consider feature engineering or hyperparameter tuning")
        
        return recommendations

    def _generate_output_recommendations(self, issues: List) -> List[str]:
        """Generate output quality recommendations."""
        recommendations = []
        
        if any(i.get("type") == "missing_field" for i in issues):
            recommendations.append("Include all required fields in response")
        
        if any(i.get("type") == "poor_error_handling" for i in issues):
            recommendations.append("Improve error reporting with detailed messages")
        
        return recommendations

    def _ai_code_review(self, code_content: str, agent_name: str = "unknown") -> Dict[str, Any]:
        """AI-powered contextual code review using Ollama."""
        try:
            prompt = f"""
Analyze this {agent_name} Python code for issues and improvements:

Code:
```python
{code_content[:1000]}
```

Return JSON: {{"issues": [], "suggestions": [], "score": 75}}"""
            
            ai_response = self._query_llm(prompt)
            
            return {
                "ai_issues": ai_response.get("issues", []),
                "ai_suggestions": ai_response.get("suggestions", []),
                "ai_quality_score": ai_response.get("score", 70),
                "llm_model": f"ollama-{OLLAMA_CONFIG.get('model', 'llama2')}",
                "analysis_context": agent_name
            }
        except Exception as e:
            logger.warning(f"AI review failed: {str(e)}")
            return {
                "ai_issues": [],
                "ai_suggestions": ["AI review unavailable - check Ollama service"],
                "ai_quality_score": 0,
                "error": str(e)
            }
    
    def _query_llm(self, prompt: str) -> Dict[str, Any]:
        """Query Ollama LLM using existing configuration."""
        try:
            import requests
            
            # Use existing Ollama configuration
            ollama_url = f"{OLLAMA_CONFIG['base_url']}/api/generate"
            
            payload = {
                "model": OLLAMA_CONFIG.get("model", "llama2"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": OLLAMA_CONFIG.get("temperature", 0.1),
                    "top_p": OLLAMA_CONFIG.get("top_p", 0.9),
                    "max_tokens": OLLAMA_CONFIG.get("max_tokens", 500)
                }
            }
            
            response = requests.post(ollama_url, json=payload, timeout=OLLAMA_CONFIG.get("timeout", 30))
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result.get("response", "")
                
                # Parse LLM response (expecting JSON format)
                try:
                    import json
                    parsed_response = json.loads(llm_output)
                    return parsed_response
                except json.JSONDecodeError:
                    return self._fallback_response()
            else:
                logger.warning(f"Ollama API error: {response.status_code}")
                return self._fallback_response()
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ollama connection failed: {str(e)}")
            return self._fallback_response()
        except Exception as e:
            logger.error(f"LLM query failed: {str(e)}")
            return self._fallback_response()
    
    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when LLM is unavailable."""
        return {
            "issues": [{
                "type": "llm_unavailable",
                "line": 0,
                "message": "AI review unavailable - using static analysis only",
                "severity": "info"
            }],
            "suggestions": [
                "Ensure Ollama is running on localhost:11434",
                "Check network connectivity",
                "Verify model availability"
            ],
            "score": 0
        }

    def _generate_code_fixes(self, issues: List[Dict]) -> Dict[str, str]:
        """Generate actual code fixes."""
        fixes = {}
        for issue in issues:
            issue_type = issue.get("type", "")
            line = issue.get("line", 0)
            
            if issue_type == "dangerous_function":
                if issue.get("function") == "eval":
                    fixes[f"line_{line}"] = "Replace eval() with ast.literal_eval()"
                elif issue.get("function") == "exec":
                    fixes[f"line_{line}"] = "Avoid exec() - use proper function calls"
            elif issue_type == "complex_function":
                fixes[f"line_{line}"] = f"Break down {issue.get('function')}() into smaller functions"
        return fixes

    def _analyze_documentation(self, code_content: str) -> List[Dict[str, Any]]:
        """Analyze documentation quality."""
        issues = []
        try:
            tree = ast.parse(code_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        issues.append({
                            "type": "missing_docstring",
                            "function": node.name,
                            "line": node.lineno,
                            "severity": "medium",
                            "location": str(node.lineno)
                        })
        except SyntaxError:
            pass
        return issues



    def _detect_code_smells(self, code_content: str) -> List[Dict[str, Any]]:
        """Detect code smells and anti-patterns."""
        smells = []
        
        # Long parameter lists
        if re.search(r'def\s+\w+\([^)]{50,}\)', code_content):
            smells.append({
                "type": "long_parameter_list",
                "message": "Function has too many parameters",
                "severity": "medium"
            })
        return smells
        
    def _standard_response(self, status: str, action: str, data: Optional[Dict] = None, errors: Optional[List] = None) -> Dict[str, Any]:
        """Generate standardized response with strict schema."""
        return {
            "status": status,
            "action": action,
            "data": data or {},
            "errors": errors or []
        }






            


        
    

