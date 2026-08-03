"""
Defines models related to traffic in surrogacy.
"""


from typing import NewType

TimeSFCRequests = NewType("TimeSFCRequests", list[dict[str, float]])
