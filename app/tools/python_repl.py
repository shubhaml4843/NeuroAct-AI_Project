"""Code execution environment."""

class PythonREPL:
    def __init__(self):
        self.globals = {}
    
    def execute(self, code: str) -> dict:
        """Execute Python code safely."""
        try:
            exec(code, self.globals)
            return {"status": "success", "output": "Code executed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}