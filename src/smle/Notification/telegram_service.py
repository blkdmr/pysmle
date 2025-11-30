from abc import ABC, abstractmethod
import os
import requests
from .service import Service
from typing import Literal


class TelegramBot(ABC):
    @abstractmethod
    def send_smle_notification(self, message: str) -> None:
        pass

class Telegram(Service):
    """Telegram notification service using webhooks."""
    
    def __init__(
        self,
        bot_token: str="",
        chat_id: str="",
        parse_mode: Literal["Markdown", "HTML"] | None=None,
        bot: TelegramBot | None=None,
    ):
        """Initialize Telegram service.
        
        Args:
            bot_token: Telegram bot token. If None, reads from TELEGRAM_BOT_TOKEN env var.
            chat_id: Telegram chat ID. If None, reads from TELEGRAM_CHAT_ID env var.
            
        Raises:
            ValueError: If webhook URL is not provided or found in environment.
        """
        super().__init__()
        self._bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN") or self._dotenv.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID") or self._dotenv.get("TELEGRAM_CHAT_ID")
        self._parse_mode = parse_mode
        self._bot = bot
        if not bot and (not self._bot_token or not self._chat_id):
            raise ValueError(
                "Telegram bot token and chat ID must be provided or set in TELEGRAM_BOT_TOKEN "
                "and TELEGRAM_CHAT_ID environment variables"
            )
        self._telegram_api_url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"

    def send_notification(self, message: str) -> None:
        """Send notification to Telegram chat.
        
        Args:
            message: The message to send.
            
        Raises:
            requests.exceptions.RequestException: If the request fails.
        """
        if self._bot:
            self._bot.send_smle_notification(message)
            return
        
        params = {}
        if self._parse_mode:
            params["parse_mode"] = self._parse_mode
        data = {
            "chat_id": self._chat_id,
            "text": message,
            "disable_notification": True,
        }

        try:
            result = requests.post(self._telegram_api_url, json=data, params=params, timeout=10)
            result.raise_for_status()
        except requests.exceptions.RequestException as err:
            raise requests.exceptions.RequestException(f"Failed to send Telegram notification: {err}")
