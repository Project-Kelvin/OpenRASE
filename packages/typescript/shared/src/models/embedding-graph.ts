export interface ChainEntity {
    id: string;
    ip?: string;
}

export interface VNFEntity extends ChainEntity {
    name?: string;
}

export interface ForwardingLink {
    source: ChainEntity;
    destination: ChainEntity;
    links: string[];
}

export interface VNF {
    host: ChainEntity;
    vnf: VNFEntity;
    next: VNF | VNF[] | string | string[];
    isTraversed?: boolean;
}

export type VNFUpdated = Pick<Partial<VNF>, "next"> & Omit<VNF, "next">;

export interface EmbeddingGraph {
    sfcID: string;
    vnfs: VNF;
    links: ForwardingLink[];
    isTraversed?: boolean;
}

export type EmbeddingGraphs = EmbeddingGraph[];
