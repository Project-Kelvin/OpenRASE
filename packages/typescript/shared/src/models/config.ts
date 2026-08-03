export interface SFFNetwork {
    networkIP: string;
    sffIP: string;
    hostIP: string;
    mask: string;
}

export interface SFF {
    network1: SFFNetwork;
    network2: SFFNetwork;
    port: number;
    txPort: number;
}

export interface Server {
    port: number;
}

export interface VNFProxy {
    port: number;
}

export interface SFCClassifier {
    port: number;
}

export interface General {
    requestTimeout: number;
}

export interface IPRange {
    mask: number;
}

export interface VNFs {
    sharedVolumes: {
        [ key: string ]: string;
    };
    names: string[];
}

export interface K6 {
    vus: number;
    startRate: number;
    timeUnit: string;
    executor: string;
}

export interface Config {
    sff: SFF;
    server: Server;
    vnfProxy: VNFProxy;
    sfcClassifier: SFCClassifier;
    general: General;
    repoAbsolutePath: string;
    templates: string[];
    ipRange: IPRange;
    vnfs: VNFs;
    k6: K6;
}
