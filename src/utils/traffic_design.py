"""
This defines utils associated with traffic design.
"""

from shared.models.traffic_design import TrafficDesign


def generateTrafficDesign(dataFile: str, scale: int = 1) -> "TrafficDesign":
    """
    Generate the Traffic Design.

    Parameters:
        dataFile (str): the CSV data file.
        scale (int): the scale factor.

    Returns:
        TrafficDesign: the Traffic Design.
    """

    design: TrafficDesign = []

    with open(dataFile, "r", encoding="utf8") as file:
        _header = file.readline()  # Read and discard the header
        for line in file:
            numOfReqs: int = int(line)
            # scale by 1
            numOfReqs = numOfReqs * scale
            #Considering an hour is equal to 4 seconds
            rate: int = numOfReqs // 4
            design.append({
                "target": rate,
                "duration": '4s'
            })

    return design

def calculateTrafficDuration(trafficDesign: TrafficDesign) -> int:
    """
    Calculate the duration of the traffic.

    Parameters:
        trafficDesign (TrafficDesign): The traffic design.

    Returns:
        int: The duration of the traffic.
    """

    totalDuration: int = 0

    for traffic in trafficDesign:
        durationText: str = traffic["duration"]
        unit: str  = durationText[-1]
        if unit == "s":
            totalDuration += int(durationText[:-1])
        elif unit == "m":
            totalDuration += int(durationText[:-1]) * 60
        elif unit == "h":
            totalDuration += int(durationText[:-1]) * 60 * 60

    return totalDuration
