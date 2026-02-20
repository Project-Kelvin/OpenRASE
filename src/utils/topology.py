"""
This generates topoigies.
"""

from shared.models.topology import Host, Link, Switch, Topology

from constants.topology import SERVER, SFCC


def generateFatTreeTopology(k: int, bandwidth: int, cpu: int, memory: int, delay: int = None) -> Topology:
    """
    Generate a Fat Tree Topology.

    Parameters:
        k (int): the number of ports in a switch. Should be an even number.
        bandwidth (int): the bandwidth of the links in Mbit/s.
        cpu (int): the CPU of the hosts.
        memory (int): the memory of the hosts.
        delay (int): the delay of the links in ms.

    Returns:
        Topology: the generated Fat Tree Topology.
    """

    if k % 2 != 0:
        raise ValueError("k should be an even number")

    coreSwNo: int = (k // 2) ** 2
    podsNo: int = k
    aggrSwPerPodNo: int = k // 2
    edgeSwPerPodNo: int = k // 2
    aggrSwLinkNo: int = k // 2
    hostsPerEdgeSw: int = k // 2
    totalHosts: int = k**3 // 4

    hosts: "list[Host]" = [{"id": f"h{i}", "cpu": cpu, "memory": memory} for i in range(1, (totalHosts + 1 - 2))]
    coreSwitches: "list[Switch]" = [{ "id": f"cs1{i}" } for i in range(1, coreSwNo + 1)]
    aggrSwitches: "list[Switch]" = [{ "id": f"as2{i}" } for i in range(1, (aggrSwPerPodNo * podsNo) + 1)]
    edgeSwitches: "list[Switch]" = [{ "id": f"es3{i}" } for i in range(1, (edgeSwPerPodNo * podsNo) + 1)]

    links: "list[Link]" = []
    pods: "list[dict[str, list[str]]]" = []

    for i, sw in enumerate(edgeSwitches):
        if i == 0:
            links.append(Link(source=sw["id"], destination=SFCC, bandwidth=bandwidth))
        elif i == len(edgeSwitches) - 1:
            links.append(Link(source=sw["id"], destination=SERVER, bandwidth=bandwidth))

        begin: int = i * hostsPerEdgeSw
        end: int = (i + 1) * hostsPerEdgeSw - 1

        if i != 0:
            begin -= 1

        for host in hosts[begin : end]:
            links.append(
                Link(source=sw["id"], destination=host["id"], bandwidth=bandwidth)
            )

    for pod in range(podsNo):
        pods.append({
            "aggr": aggrSwitches[pod * aggrSwPerPodNo: (pod + 1) * aggrSwPerPodNo],
            "edge": edgeSwitches[pod * edgeSwPerPodNo: (pod + 1) * edgeSwPerPodNo]
        })

    for pod in pods:
        for i, aggr in enumerate(pod["aggr"]):
            for edge in pod["edge"]:
                links.append(
                    Link(source=aggr["id"], destination=edge["id"], bandwidth=bandwidth, delay=delay)
                )

            for sw in coreSwitches[i * aggrSwLinkNo: (i+1) * aggrSwLinkNo]:
                links.append(
                    Link(source=sw["id"], destination=aggr["id"], bandwidth=bandwidth, delay=delay)
                )

    return Topology(hosts=hosts, switches=coreSwitches + aggrSwitches + edgeSwitches, links=links)

def generateTopologyFromEdgeList(edgeListFile: str, cpus: int, memory: int, bandwidth: int, delay: int = None) -> Topology:
    """
    Generate a topology from an edge list.

    Parameters:
        edgeListFile (str): the path to the edge list file.
        cpus (int): the CPU of the hosts.
        memory (int): the memory of the hosts.
        bandwidth (int): the bandwidth of the links in Mbit/s.
        delay (int): the delay of the links in ms.

    Returns:
        Topology: the generated topology.
    """

    hostIDs: dict[int, Host] = {}
    switchIDs: dict[int, Switch] = {}
    linkIDs: list[str] = []
    links: list[Link] = []
    hosts: list[Host] = []
    switches: list[Switch] = []
    topology: Topology = {}


    with open(edgeListFile, "r") as f:
        i: int = 0
        firstSwitchID: int = 0
        lastSwitchID: int = 0

        for line in f:
            components: list[str] = line.split(" ")

            if i == 0:
                firstSwitchID = int(components[0])

            lastSwitchID = int(components[0])

            if int(components[0]) not in hostIDs:
                host: Host = Host(
                    id=f"h{int(components[0])}",
                    cpu=cpus,
                    memory=memory,
                )
                switch: Switch = Switch(id=f"s{int(components[0])}")
                hostIDs[int(components[0])] = host
                switchIDs[int(components[0])] = switch
                hosts.append(host)
                switches.append(switch)
                links.append(
                    Link(
                        source=f"h{int(components[0])}",
                        destination=f"s{int(components[0])}",
                        bandwidth=bandwidth,
                        delay=delay,
                    )
                )
            if int(components[1]) not in hostIDs:
                host: Host = Host(
                    id=f"h{int(components[1])}",
                    cpu=cpus,
                    memory=memory,
                )
                switch: Switch = Switch(id=f"s{int(components[1])}")
                hostIDs[int(components[1])] = host
                switchIDs[int(components[1])] = switch
                hosts.append(host)
                switches.append(switch)
                links.append(
                    Link(
                        source=f"h{int(components[1])}",
                        destination=f"s{int(components[1])}",
                        bandwidth=bandwidth,
                        delay=delay,
                    )
                )
            if (
                f"{components[0]}-{components[1]}" not in linkIDs
                and f"{components[1]}-{components[0]}" not in linkIDs
            ):
                linkIDs.append(f"{components[0]}-{components[1]}")
                links.append(
                    Link(
                        source=f"s{int(components[0])}",
                        destination=f"s{int(components[1])}",
                        bandwidth=bandwidth,
                        delay=delay,
                    )
                )
            i += 1

        serverSwitchID: str = "91"
        sfccSwitchID: str = "92"
        switches.append(Switch(id=f"s{serverSwitchID}"))
        switches.append(Switch(id=f"s{sfccSwitchID}"))

        links.append(
            Link(
                source=SERVER,
                destination=f"s{serverSwitchID}",
                bandwidth=bandwidth,
                delay=delay,
            )
        )
        links.append(
            Link(
                source=f"s{serverSwitchID}",
                destination=f"s{lastSwitchID}",
                bandwidth=bandwidth,
                delay=delay,
            )
        )
        links.append(
            Link(
                source=SFCC,
                destination=f"s{sfccSwitchID}",
                bandwidth=bandwidth,
                delay=delay,
            )
        )
        links.append(
            Link(
                source=f"s{sfccSwitchID}",
                destination=f"s{firstSwitchID}",
                bandwidth=bandwidth,
                delay=delay,
            )
        )
        topology["hosts"] = hosts
        topology["switches"] = switches
        topology["links"] = links

    return topology
