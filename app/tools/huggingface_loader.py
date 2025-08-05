"""HuggingFace model and embedding loader."""

class HuggingFaceLoader:
    def __init__(self):
        self.loaded_models = {}
    
    def load_model(self, model_name: str) -> dict:
        """Load model from HuggingFace."""
        self.loaded_models[model_name] = f"model_{model_name}"
        return {"status": "loaded", "model_id": model_name}
    
    def load_tokenizer(self, model_name: str) -> dict:
        """Load tokenizer from HuggingFace."""
        return {"status": "loaded", "tokenizer_id": model_name}