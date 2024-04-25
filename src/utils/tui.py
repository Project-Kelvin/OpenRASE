from threading import Thread
from time import sleep
from textual.app import App, ComposeResult
from textual.widgets import Static, Button
from textual.reactive import reactive
from textual.containers import ScrollableContainer, Horizontal
from textual.screen import Screen

class Logger(Static):
    """
    Displays a log message
    """

    log = reactive("", layout=True)

    def render(self) -> str:
        """
        Render the log message.

        Returns:
            str: The log message.
        """

        return self.log

class LoggerContainer(Static):
    """
    A container that displays a series of log messages.
    """

    def compose(self) -> ComposeResult:
        """
        Compose the container.

        Returns:
            ComposeResult: The container.
        """

        yield ScrollableContainer(
            Logger()
        )

    def appendToLog(self, msg: str)-> None:
        """
        Append a log message
        """

        self.query_one(Logger).log += msg

    def addToLog(self, msg: str) -> None:
        """
        Log a message.
        """

        self.query_one(Logger).log = msg

class LoggerScreen(Screen):
    """
    Displays two loggers split horizontally across the screen.
    """

    logger: LoggerContainer = None
    solverLogger: LoggerContainer = None

    def compose(self) -> ComposeResult:
        """
        Compose the logger screen.

        Returns:
            ComposeResult: The logger screen.
        """

        self.logger = LoggerContainer(id="loggerContainer")
        self.solverLogger = LoggerContainer(id="solverLoggerContainer")
        yield Horizontal(
            self.logger,
            self.solverLogger
        )

    # pylint: disable=invalid-name
    def on_mount(self) -> None:
        """
        Mount the logger screen.
        """

        self.logger.styles.width = "50%"
        self.solverLogger.styles.width = "50%"

    def appendToLog(self, msg: str) -> None:
        """
        Append a message to the logger.
        """

        self.logger.appendToLog(msg)

    def appendToSolverLog(self, msg: str) -> None:
        """
        Append a message to the solver logger.
        """

        self.solverLogger.appendToLog(msg)

    def addToLog(self, msg: str) -> None:
        """
        Log a message.
        """

        self.logger.addToLog(msg)

    def addToLogSolver(self, msg: str) -> None:
        """
        Log a message.
        """

        self.solverLogger.addToLog(msg)

class SplashScreen(Screen):
    """
    Displays the splash screen.
    """

    def compose(self) -> ComposeResult:
        """
        Yield the splash screen.

        Returns:
            ComposeResult: The splash screen.
        """

        yield Static("[bold blue]OpenRASE[/]\n[italic dim]Loading...[/]")


class UI(App):
    """
    The main application.
    """

    title="OpenRASE"

    # pylint: disable=invalid-name
    def on_mount(self) -> None:
        """
        Mount the application.
        """

        self.install_screen(SplashScreen(), name="splash")
        self.install_screen(LoggerScreen(), name="logger")
        self.push_screen("logger")
        self.push_screen("splash")

    def addToLog(self, msg: str) -> None:
        """
        Log a message.

        Parameters:
            msg (str): The message to log.
        """

        self.get_screen("logger").addToLog(msg)

    def addToLogSolver(self, msg: str) -> None:
        """
        Log a message.

        Parameters:
            msg (str): The message to log.
        """

        self.get_screen("logger").addToLogSolver(msg)

    def appendtoLog(self, msg: str) -> None:
        """
        Append to the log.

        Parameters:
            msg (str): The message to append.
        """

        self.get_screen("logger").appendToLog(msg)

    def appendToSolverLog(self, msg: str) -> None:
        """
        Append to the solver log.

        Parameters:
            msg (str): The message to append.
        """

        self.get_screen("logger").appendToSolverLog(msg)

class TUI():
    """
    The TUI class is responsible for displaying the progress of OpenRASE.
    """

    _isInitialized: bool = False
    _disable: bool = False

    @classmethod
    def init(cls) -> None:
        """
        Initialize the TUI.
        """


        cls._isInitialized = False

        if cls._disable:
            return

        cls.app: UI = UI()

        def switchToLogger() -> None:
            sleep(2)
            cls.app.switch_screen("logger")
            cls._isInitialized = True

        Thread(target=switchToLogger).start()

        cls.app.run()

    @classmethod
    def _wait(cls) -> None:
        """
        Wait for the TUI to be initialized.
        """

        while not cls._isInitialized:
            pass

    @classmethod
    def log(cls, text: str) -> None:
        """
        Log a message.

        Parameters:
            text (str): The text to log.
        """

        if cls._disable:
            return

        cls._wait()
        cls.app.addToLog(text)

    @classmethod
    def logSolver(cls, text: str) -> None:
        """
        Log a message.
        """

        if cls._disable:
            return

        cls._wait()
        cls.app.addToLogSolver(text)

    @classmethod
    def appendToLog(cls, text: str, error: bool = False) -> None:
        """
        Append to the log.

        Parameters:
            text (str): The text to append.
            error (bool): Whether the text is an error.
        """

        if cls._disable:
            return

        cls._wait()
        cls.app.appendtoLog(f"\n[{'red' if error else 'green'}]{text}[/]")

    @classmethod
    def appendToSolverLog(cls, text: str, error: bool = False) -> None:
        """
        Append to the solver log.

        Parameters:
            text (str): The text to append.
            error (bool): Whether the text is an error.
        """

        if cls._disable:
            return

        cls._wait()
        cls.app.appendToSolverLog(f"\n[{'red' if error else 'green'}]{text}[/]")

    @classmethod
    def exit(cls) -> None:
        """
        Exit the TUI.
        """

        if cls._disable:
            return

        cls._isInitialized = False
        cls.app.exit()
        cls.app.uninstall_screen("logger")
        cls.app.uninstall_screen("splash")

    @classmethod
    def disable(cls) -> None:
        """
        Disable the TUI.
        """

        cls._disable = True
