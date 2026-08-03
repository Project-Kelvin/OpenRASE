"""
Defines a class to handle publishing and subscribing to notifications.
Additionally, defines an abstract class for subscribers.
"""

from abc import ABC, abstractmethod
from typing import Any


class NotificationSystem():
    """
    Class to handle publishing and subscribing to notifications.
    """

    _topics: "dict[str, list[object]]" = {}

    @classmethod
    def publish(cls, topic: str, *args: "list[Any]") -> None:
        """
        Publish a notification.

        Parameters:
            topic (str): The topic of the notification.
            args (list[Any]): The arguments of the notification.
        """

        if topic in cls._topics and cls._topics[topic] is not None:
            for sub in cls._topics[topic]:
                sub.receiveNotification(topic, *args)

    @classmethod
    def subscribe(cls, topic: str, subscriber: object) -> None:
        """
        Subscribe to a topic.

        Parameters:
            topic (str): The topic to subscribe to.
            subscriber (object): The subscriber.
        """

        if topic in cls._topics:
            cls._topics[topic].append(subscriber)
        else:
            cls._topics[topic] = [subscriber]

    @classmethod
    def unsubscribeAll(cls) -> None:
        """
        Unsubscribe all subscribers.
        """

        cls._topics = {}

class Subscriber(ABC):
    """
    Abstract class for subscribers.
    """

    @abstractmethod
    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        """
        Receive a notification.

        Parameters:
            topic (str): The topic of the notification.
            args (list[Any]): The arguments of the notification.
        """
