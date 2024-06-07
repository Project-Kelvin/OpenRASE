import { VNF, VNFUpdated } from "../models/embedding-graph";

/**
 * This function encodes the Embedding Graph to a base64 string.
 *
 * @param data The Embedding Graph to be encoded.
 * @returns base64 encoded string
 */
export function sfcEncode(data: VNF | VNF[] | VNFUpdated[]): string {

    const jsonData: string = JSON.stringify(data);
    const base64Data: string = btoa(jsonData)

    return base64Data;
}

/**
 * This function decodes the base64 encoded string to an Embedding Graph.
 *
 * @param data The base64 encoded string to be decoded.
 * @returns The decoded Embedding Graph.
 */
export function sfcDecode(data: string): VNF | VNF[] | VNFUpdated[] {
    const jsonString: string = atob(data);
    const eg: VNF = JSON.parse(jsonString);

    return eg;
}
