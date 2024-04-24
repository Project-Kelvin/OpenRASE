"""
This defines utils associated with traffic design.
"""

from shared.models.traffic_design import TrafficDesign


def generateTrafficDesign(dataFile: str) -> "TrafficDesign":
    """
    Generate the Traffic Design.

    Parameters:
        dataFile (str): the CSV data file.

    Returns:
        TrafficDesign: the Traffic Design.
    """

    design: TrafficDesign = []

    with open(dataFile, "r", encoding="utf8") as file:
        _header = file.readline()  # Read and discard the header
        for line in file:
            numOfReqs: int = int(line)
            # scale by 10
            numOfReqs = numOfReqs * 10
            #Considering an hour is equal to 4 seconds
            rate: int = numOfReqs // 4
            design.append({
                "target": rate,
                "duration": '4s'
            })

    return design
