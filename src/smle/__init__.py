import traceback
import sys
import os
from colorama import Fore, Style

from smle.args import Parser
from smle.logging import Logger
from smle.utils import generate_haiku_id
from smle.secrets.keystore import KeyStore

class SMLE:

    """
    The base SMLE application
    """

    def __init__(self, config_file: str="smle.yaml"):
        self._session_id = generate_haiku_id()
        self._entrypoint_fn = None
        self._parser: Parser | None = None
        self._config_file = config_file

    @property
    def config_file(self):
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        """
        The method to set the YAML file.
        """
        self._config_file = config_file

        self._keystore = KeyStore()

        #webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        #self._notifier = Notifier(webhook_url)

    def entrypoint(self, entrypoint_fn):
        """
        The decorator to register the main execution function.
        """
        self._entrypoint_fn = entrypoint_fn
        return entrypoint_fn

    def run(self):
        """
        The method that executes the application's core logic.
        """

        if not self._entrypoint_fn:
            print(f"{Fore.RED}[SMLE] No main function registered. {Style.RESET_ALL}")
            print(f"{Fore.RED}[SMLE] Please use {Fore.LIGHTYELLOW_EX}@app.entrypoint{Fore.RED} to register your main function{Style.RESET_ALL}")
            sys.exit(1)

        self._parser = Parser()
        self._parser.config_file = self._config_file
        self._args = self._parser.load_configuration()
        self._args["session_id"] = self._session_id
        self._logger = Logger(self._args)
        self._logger.start()

        self._parser.print()

        try:
            # The execution of the decorated user function
            print(f"{Fore.GREEN}[SMLE] Application starting from {Fore.LIGHTYELLOW_EX}{self._entrypoint_fn.__name__}{Fore.GREEN} entrypoint.{Style.RESET_ALL}")

            #self._notifier.send_notification("start")
            result = self._entrypoint_fn(self._args)
            #self._notifier.send_notification("end")

            return result
        except Exception:
            # Print the traceback on failure
            #self._notifier.send_notification(traceback.format_exc())
            print(traceback.format_exc())
            sys.exit(1)
        finally:
            self._logger.stop()