import requests

class Notifier:
    def __init__(self, webhook_url):
        self._discord_webhook_url = webhook_url

    def send_notification(self, message):
        self._send_discord_notification(message)

    def _send_discord_notification(self, message):

        data = {
            "content": message,
            "username": "pysmle"
        }

        result = requests.post(self._discord_webhook_url, json=data)

        try:
            result.raise_for_status()
        except requests.exceptions.HTTPError as err:
            print(f"Error: {err}")