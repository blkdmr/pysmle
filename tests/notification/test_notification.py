"""Tests for the notification system."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from smle.Notification import (
    Notifier,
    Service,
    Discord,
    Slack,
    Email,
    Telegram,
    TelegramBot,
)


class MockService(Service):
    """Mock service for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.messages_sent = []
    
    def send_notification(self, message: str) -> None:
        if self.should_fail:
            raise Exception("Mock service failure")
        self.messages_sent.append(message)


class TestNotifier:
    """Test cases for Notifier class."""
    
    def test_init(self):
        """Test notifier initialization."""
        notifier = Notifier()
        assert notifier.services == []
        assert notifier.get_services() == []
    
    def test_add_service(self):
        """Test adding services."""
        notifier = Notifier()
        service = MockService()
        
        notifier.add_service(service)
        
        assert len(notifier.services) == 1
        assert service in notifier.services
        assert notifier.get_services() == ["MockService"]
    
    def test_remove_service(self):
        """Test removing services."""
        notifier = Notifier()
        service = MockService()
        
        notifier.add_service(service)
        notifier.remove_service(service)
        
        assert len(notifier.services) == 0
        assert notifier.get_services() == []
    
    def test_remove_nonexistent_service(self):
        """Test removing a service that doesn't exist."""
        notifier = Notifier()
        service = MockService()
        
        with pytest.raises(ValueError):
            notifier.remove_service(service)
    
    def test_notify_success(self):
        """Test successful notification to all services."""
        notifier = Notifier()
        service1 = MockService()
        service2 = MockService()
        
        notifier.add_service(service1)
        notifier.add_service(service2)
        
        message = "Test message"
        notifier.notify(message)
        
        assert service1.messages_sent == [message]
        assert service2.messages_sent == [message]
    
    def test_notify_with_failures(self):
        """Test notification with some services failing."""
        notifier = Notifier()
        good_service = MockService()
        bad_service = MockService(should_fail=True)
        
        notifier.add_service(good_service)
        notifier.add_service(bad_service)
        
        message = "Test message"
        notifier.notify(message)  # Should not raise exception
        
        assert good_service.messages_sent == [message]
        assert bad_service.messages_sent == []
    
    def test_notify_no_services(self):
        """Test notification with no services registered."""
        notifier = Notifier()
        notifier.notify("Test message")  # Should not raise exception
    
    def test_clear_services(self):
        """Test clearing all services."""
        notifier = Notifier()
        service1 = MockService()
        service2 = MockService()
        
        notifier.add_service(service1)
        notifier.add_service(service2)
        notifier.clear_services()
        
        assert len(notifier.services) == 0
        assert notifier.get_services() == []


class TestDiscord:
    """Test cases for Discord service."""
    
    @patch.dict(os.environ, {"DISCORD_WEBHOOK": "https://discord.com/webhook/test"})
    def test_init_with_env_var(self):
        """Test Discord initialization with environment variable."""
        discord = Discord()
        assert discord._discord_webhook_url == "https://discord.com/webhook/test"
    
    def test_init_with_url(self):
        """Test Discord initialization with direct URL."""
        url = "https://discord.com/webhook/test"
        discord = Discord(webhook_url=url)
        assert discord._discord_webhook_url == url
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_url(self):
        """Test Discord initialization without URL."""
        with pytest.raises(ValueError, match="Discord webhook URL must be provided"):
            Discord()
    
    @patch('smle.Notification.discord_service.requests.post')
    def test_send_notification_success(self, mock_post):
        """Test successful Discord notification."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        discord = Discord(webhook_url="https://discord.com/webhook/test")
        discord.send_notification("Test message")
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://discord.com/webhook/test"
        assert kwargs['json']['content'] == "Test message"
        assert kwargs['json']['username'] == "pysmle"
    
    @patch('smle.Notification.discord_service.requests.post')
    def test_send_notification_failure(self, mock_post):
        """Test Discord notification failure."""
        mock_post.side_effect = Exception("Network error")
        
        discord = Discord(webhook_url="https://discord.com/webhook/test")
        
        with pytest.raises(Exception):
            discord.send_notification("Test message")


class TestTelegram:
    """Test cases for Telegram service."""
    
    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "1234567890",
        "TELEGRAM_CHAT_ID": "1234567890"
    })
    def test_init_with_env_var(self):
        """Test Telegram initialization with environment variable."""
        telegram = Telegram()
        assert telegram._bot_token == "1234567890"
        assert telegram._chat_id == "1234567890"
        assert telegram._parse_mode is None
        assert telegram._bot is None
        assert telegram._telegram_api_url == "https://api.telegram.org/bot1234567890/sendMessage"
    
    def test_init_with_bot_and_chat_id(self):
        """Test Telegram initialization with direct URL."""
        chat_id = "1234567890"
        bot_token = "1234567890"
        telegram = Telegram(bot_token=bot_token, chat_id=chat_id)
        assert telegram._bot_token == bot_token
        assert telegram._chat_id == chat_id
        assert telegram._parse_mode is None
        assert telegram._bot is None
        assert telegram._telegram_api_url == "https://api.telegram.org/bot1234567890/sendMessage"
    

    def test_init_with_bot(self):
        """Test Telegram initialization with direct URL."""
        chat_id = "1234567890"
        bot_token = "1234567890"
        class MockBot(TelegramBot):
            def send_smle_notification(self, message: str) -> None:
                pass
        bot = MockBot()
        telegram = Telegram(bot=bot)
        assert telegram._bot == bot
        assert telegram._telegram_api_url == "https://api.telegram.org/botNone/sendMessage"
        assert telegram._parse_mode is None
        assert telegram._bot_token is None
        assert telegram._chat_id is None
        with patch.object(bot, 'send_smle_notification') as mock_send_smle_notification:
            telegram.send_notification("Test message")
            mock_send_smle_notification.assert_called_once_with("Test message")

    def test_init_with_parse_mode(self):
        """Test Telegram initialization with parse mode."""
        telegram = Telegram(bot_token="1234567890", chat_id="1234567890", parse_mode="Markdown")
        assert telegram._parse_mode == "Markdown"
    
    @patch('smle.Notification.discord_service.requests.post')
    def test_send_notification_success(self, mock_post):
        """Test successful Telegram notification."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        telegram = Telegram(bot_token="1234567890", chat_id="1234567890", parse_mode="Markdown")
        telegram.send_notification("Test message")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.telegram.org/bot1234567890/sendMessage"
        assert kwargs['json']['chat_id'] == "1234567890"
        assert kwargs['json']['text'] == "Test message"
        assert kwargs['params']['parse_mode'] == "Markdown"

class TestSlack:
    """Test cases for Slack service."""
    
    @patch.dict(os.environ, {"SLACK_WEBHOOK": "https://hooks.slack.com/test"})
    def test_init_with_env_var(self):
        """Test Slack initialization with environment variable."""
        slack = Slack()
        assert slack._webhook_url == "https://hooks.slack.com/test"
    
    def test_init_with_url(self):
        """Test Slack initialization with direct URL."""
        url = "https://hooks.slack.com/test"
        slack = Slack(webhook_url=url)
        assert slack._webhook_url == url
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_url(self):
        """Test Slack initialization without URL."""
        with pytest.raises(ValueError, match="Slack webhook URL must be provided"):
            Slack()


class TestEmail:
    """Test cases for Email service."""
    
    @patch.dict(os.environ, {
        "SMTP_SERVER": "smtp.gmail.com",
        "SMTP_USERNAME": "test@example.com",
        "SMTP_PASSWORD": "password",
        "SMTP_FROM_EMAIL": "test@example.com",
        "SMTP_TO_EMAILS": "recipient@example.com"
    })
    def test_init_with_env_vars(self):
        """Test Email initialization with environment variables."""
        email = Email()
        assert email.smtp_server == "smtp.gmail.com"
        assert email.username == "test@example.com"
        assert email.to_emails == ["recipient@example.com"]
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_config(self):
        """Test Email initialization without configuration."""
        with pytest.raises(ValueError, match="Email service requires"):
            Email()
    
    @patch('smle.Notification.email_service.smtplib.SMTP')
    def test_send_notification_success(self, mock_smtp):
        """Test successful email notification."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        email = Email(
            smtp_server="smtp.test.com",
            username="test@example.com", 
            password="password",
            from_email="test@example.com",
            to_emails=["recipient@example.com"]
        )
        
        email.send_notification("Test message")
        
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("test@example.com", "password")
        mock_server.send_message.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
