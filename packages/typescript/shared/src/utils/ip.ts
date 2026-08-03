import ipRangeCheck from "ip-range-check";

export const checkIPBelongsToNetwork: (ip: string, networkIP: string) => boolean = (ip: string, networkIP: string): boolean => {
    return ipRangeCheck(ip, networkIP);
};
