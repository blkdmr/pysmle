from abc import ABC, abstractmethod
from dotenv import dotenv_values

class Service(ABC):
    """Abstract base class for notification services."""

    def __init__(self):
        self._dotenv = dotenv_values(".env")
    
    @abstractmethod
    def send_notification(self, message: str) -> None:
        """Send a notification message.
        
        Args:
            message: The message to send.
            
        Raises:
            Exception: If the notification fails to send.
        """
        pass
