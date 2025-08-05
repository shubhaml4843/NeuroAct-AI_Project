"""JSON read/edit helpers."""
import json

class JSONEditor:
    def __init__(self):
        pass
    
    def read_json(self, file_path: str) -> dict:
        """Read JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}
    
    def write_json(self, file_path: str, data: dict) -> dict:
        """Write data to JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e)}