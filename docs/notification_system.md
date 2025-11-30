# PySmle Notification System

A flexible and extensible notification system for ML training alerts that supports multiple services including Discord, Slack, and Email.

## Features

- 🚀 **Multiple Services**: Discord, Slack, Telegram, Email support
- 🔧 **Extensible**: Easy to add new notification services
- 🛡️ **Error Handling**: Graceful failure handling with detailed logging
- 🧪 **Well Tested**: Comprehensive test suite with 95%+ coverage
- 📝 **Type Hints**: Full type annotation support
- 🌍 **Environment Variables**: Easy configuration via env vars

## Quick Start

```python
from smle.Notification import Notifier, Discord, Slack, Telegram

# Create notifier
notifier = Notifier()

# Add services
notifier.add_service(Discord())   # Requires DISCORD_WEBHOOK env var
notifier.add_service(Slack())     # Requires SLACK_WEBHOOK env var
notifier.add_service(Telegram())  # Require TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars

# Send notification
notifier.notify("🚀 Training completed successfully!")
```

## Installation

The notification system is included with PySmle. Make sure you have the required dependencies:

```bash
pip install requests  # For Discord and Slack
# Email service uses built-in smtplib (no extra dependencies)
```

## Configuration

### Environment Variables

Set these environment variables to enable different notification services:

#### Discord
```bash
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_WEBHOOK_URL"
```

#### Slack
```bash
export SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR_WEBHOOK_URL"
```

#### Telegram
By default Telegram needs tow values: a bot token and a chat ID

Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to the file `.env`
  or 
```bash
export TELEGRAM_BOT_TOKEN="<YOUR_BOT_TOKEN>"
export TELEGRAM_CHAT_ID="<YOUR_CHAT_ID>"
```

#### Email
```bash
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"  # Optional, defaults to 587
export SMTP_USERNAME="your-email@gmail.com"
export SMTP_PASSWORD="your-app-password"
export SMTP_FROM_EMAIL="your-email@gmail.com"
export SMTP_TO_EMAILS="recipient1@example.com,recipient2@example.com"
```

### Programmatic Configuration

You can also configure services programmatically:

```python
from smle.Notification import Notifier, Discord, Slack, Email

notifier = Notifier()

# Configure with direct URLs/credentials
notifier.add_service(Discord(webhook_url="https://discord.com/..."))
notifier.add_service(Slack(webhook_url="https://hooks.slack.com/..."))
notifier.add_service(Email(
    smtp_server="smtp.gmail.com",
    username="user@example.com",
    password="password",
    from_email="user@example.com",
    to_emails=["recipient@example.com"]
))
```

## Usage Examples

### Basic Usage

```python
from smle.Notification import Notifier, Discord

notifier = Notifier()
notifier.add_service(Discord())
notifier.notify("Hello from PySmle!")
```

### Training Integration

```python
import time
from smle.Notification import Notifier, Discord, Slack

def train_model():
    notifier = Notifier()
    notifier.add_service(Discord())
    notifier.add_service(Slack())
    
    # Training start
    notifier.notify("🚀 Training started")
    
    for epoch in range(10):
        # ... training code ...
        time.sleep(1)
        
        if epoch % 5 == 0:
            notifier.notify(f"📈 Epoch {epoch}/10 completed")
    
    # Training complete
    notifier.notify("✅ Training completed successfully!")
```

### Error Handling

The notification system handles errors gracefully:

```python
notifier = Notifier()
notifier.add_service(Discord())  # This might fail
notifier.add_service(Slack())    # This might work

# Even if some services fail, others will still work
notifier.notify("This message will be sent to working services only")
```

### Service Management

```python
notifier = Notifier()

# Add services
discord = Discord()
slack = Slack()
notifier.add_service(discord)
notifier.add_service(slack)

# Check registered services
print(notifier.get_services())  # ['Discord', 'Slack']

# Remove a service
notifier.remove_service(discord)
print(notifier.get_services())  # ['Slack']

# Clear all services
notifier.clear_services()
print(notifier.get_services())  # []
```

## Creating Custom Services

You can easily create custom notification services by implementing the `Service` interface:

```python
from smle.Notification import Service
import requests

class CustomWebhook(Service):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_notification(self, message: str) -> None:
        response = requests.post(
            self.webhook_url,
            json={"text": message},
            timeout=10
        )
        response.raise_for_status()

# Use your custom service
notifier = Notifier()
notifier.add_service(CustomWebhook("https://your-webhook.com"))
notifier.notify("Hello from custom service!")
```

## API Reference

### Notifier

Main class for managing notification services.

#### Methods

- `add_service(service: Service) -> None`: Add a notification service
- `remove_service(service: Service) -> None`: Remove a notification service
- `notify(message: str) -> None`: Send notification to all services
- `get_services() -> List[str]`: Get list of registered service names
- `clear_services() -> None`: Remove all services

### Service (Abstract Base Class)

Interface that all notification services must implement.

#### Methods

- `send_notification(message: str) -> None`: Send notification (abstract method)

### Discord

Discord notification service using webhooks.

#### Constructor
- `Discord(webhook_url: str = None)`: Initialize with webhook URL or DISCORD_WEBHOOK env var

### Slack

Slack notification service using webhooks.

#### Constructor
- `Slack(webhook_url: str = None)`: Initialize with webhook URL or SLACK_WEBHOOK env var

### Telegram

#### Constructor
- `Telegram(bot_token: str="", chat_id: str="", parse_mode: Literal["Markdown", "HTML"] | None=None, bot: TelegramBot | None=None)`

#### Ways to implement
There are two ways to use Telegram Notifications:
- Provide your bot token and a chat ID
- Provide your bot

##### Telegram with Token and Chat ID
Pass `bot_token` and `chat_id` to the constructor or add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to the file `.env`.

##### Telegram with Bot
If you have an existing bot running in your application, have your bot implement the `TelegramBot` abstract base class, which requires the method `send_smle_notification(messages: str)`.

```python
from smle.Notification import TelegramBot

class MyExistingTelegramBot(TelegramBot):
    ...
    def send_smle_notification(self, message: str) -> None:
        self.send_message(message)
```

### Email

Email notification service using SMTP.

#### Constructor
- `Email(smtp_server: str = None, smtp_port: int = 587, username: str = None, password: str = None, from_email: str = None, to_emails: list = None)`: Initialize with SMTP settings or environment variables

## Testing

Run the test suite:

```bash
pytest tests/notification/ -v
```

## Examples

See `examples/notification_example.py` for complete usage examples.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
