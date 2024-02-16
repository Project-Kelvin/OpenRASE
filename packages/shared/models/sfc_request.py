"""
Defines the models associated with SFC Requests.
"""




class SFCRequest(dict):
    """
    Defines the `SFCRequest` dictionary type.
    """

    sfcrID: str
    latency: int
    vnfs: "list[str]"
    strictOrder: "list[str]"
