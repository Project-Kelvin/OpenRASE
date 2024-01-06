"""
Defines a class to handle publishing and subscribing to notifications.
Additionally, defines an abstract class for subscribers.
"""

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class NotificationSystem():
    """
    Class to handle publishing and subscribing to notifications.
    """

    topics: "TypedDict[str, list[object]]" = {}

    @classmethod
    def publish(cls, topic: str, *args: "list[Any]") -> None:
        """
        Publish a notification.

        Parameters:
            topic (str): The topic of the notification.
            args (list[Any]): The arguments of the notification.
        """

        if topic in cls.topics and cls.topics[topic] is not None:
            for sub in cls.topics[topic]:
                sub.receiveNotification(topic, *args)

    @classmethod
    def subscribe(cls, topic: str, subscriber: object) -> None:
        """
        Subscribe to a topic.

        Parameters:
            topic (str): The topic to subscribe to.
            subscriber (object): The subscriber.
        """

        if topic in cls.topics:
            cls.topics[topic].append(subscriber)
        else:
            cls.topics[topic] = [subscriber]


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
