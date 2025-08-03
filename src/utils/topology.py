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
