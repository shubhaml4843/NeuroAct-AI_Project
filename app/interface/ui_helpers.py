"""UI response formatting helpers."""

class UIHelpers:
    @staticmethod
    def format_response(data: dict, format_type: str = "json") -> dict:
        """Format response for UI consumption."""
        if format_type == "json":
            return {"formatted": True, "data": data}
        return data
    
    @staticmethod
    def create_error_response(error: str, code: int = 400) -> dict:
        """Create standardized error response."""
        return {"error": True, "message": error, "code": code}