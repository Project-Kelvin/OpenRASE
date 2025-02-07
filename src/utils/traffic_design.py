"""
This defines utils associated with traffic design.
"""

from shared.models.traffic_design import TrafficDesign


def generateTrafficDesign(dataFile: str, scale: float = 1.5, hourDuration: float = 4) -> "TrafficDesign":
    """
    Generate the Traffic Design.

    Parameters:
        dataFile (str): the CSV data file.
        scale (float): the scale factor.
        hourDuration (float): the simulated duration of the hour in seconds.

    Returns:
        TrafficDesign: the Traffic Design.
    """

    design: TrafficDesign = []

    with open(dataFile, "r", encoding="utf8") as file:
        _header = file.readline()  # Read and discard the header
        for line in file:
            numOfReqs: int = int(line)
            # scale by 1
            numOfReqs = round(numOfReqs * scale)
            #Considering an hour is equal to `hourDuration` seconds
            rate: int = numOfReqs // hourDuration
            design.append({
                "target": rate,
                "duration": f'{hourDuration}s'
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
