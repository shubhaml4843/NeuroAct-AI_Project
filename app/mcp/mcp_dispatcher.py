"""Dispatcher for inter-agent communication."""

class MCPDispatcher:
    def __init__(self):
        self.agents = {}
        self.message_queue = []
    
    def register_agent(self, name: str, agent):
        """Register agent for communication."""
        self.agents[name] = agent
    
    def send_message(self, sender: str, receiver: str, message: dict) -> bool:
        """Send message between agents."""
        if receiver in self.agents:
            self.message_queue.append({"from": sender, "to": receiver, "msg": message})
            return True
        return False