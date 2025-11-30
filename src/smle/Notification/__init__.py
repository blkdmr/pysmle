"""PySmle Notification System.

A flexible notification system for ML training alerts that supports multiple services
including Discord, Slack, and Email.

Example:
    Basic usage:
    
    >>> from smle.Notification import Notifier, Discord, Slack
    >>> 
    >>> # Initialize notifier
    >>> notifier = Notifier()
    >>> 
    >>> # Add services
    >>> notifier.add_service(Discord())
    >>> notifier.add_service(Slack())
    >>> 
    >>> # Send notification
    >>> notifier.notify("Training completed successfully!")

Environment Variables:
    DISCORD_WEBHOOK: Discord webhook URL
    SLACK_WEBHOOK: Slack webhook URL
    SMTP_SERVER: SMTP server address
    SMTP_PORT: SMTP server port (default: 587)
    SMTP_USERNAME: SMTP username
    SMTP_PASSWORD: SMTP password
    SMTP_FROM_EMAIL: Sender email address
    SMTP_TO_EMAILS: Comma-separated list of recipient emails
"""

from .notifier import Notifier
from .service import Service
from .discord_service import Discord
from .slack_service import Slack
from .email_service import Email
from .telegram_service import Telegram, TelegramBot

__all__ = [
    "Notifier",
    "Service", 
    "Discord",
    "Slack",
    "Email",
    "Telegram",
    "TelegramBot"
]

__version__ = "1.0.0"
