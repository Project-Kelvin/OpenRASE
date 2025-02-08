import fastify, { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { checkIPBelongsToNetwork, getConfig, sfcDecode, sfcEncode, logger } from "shared/utils";
import { Config, VNF, VNFUpdated } from "shared/models";
import { TERMINAL, SFC_HEADER, SFC_TRAVERSED_HEADER, SFC_ID } from "shared/constants";
import http from "http";

const app: FastifyInstance = fastify({
    ignoreTrailingSlash: true
});
const config: Config = getConfig();


/**
 * Extract and validate the SFC header from the request.
 *
 * @param request The request received by the SFF.
 * @returns The SFC metadata.
 *
 * @throws If the SFC header is not found in the request.
 * @throws If the request is sent to the wrong host.
 */
function extractAndValidateSFCHeader(request: FastifyRequest): VNF {
    const sfcBase64: string = request.headers[ SFC_HEADER ] as string ?? "";

    if (!sfcBase64) {
        throw new Error(
            `The SFF could not find the SFC header ` +
            `attribute in the request from: ${ request.ip }.\n`
        );
    }

    const sfc: VNF = sfcDecode(sfcBase64) as VNF;

    return sfc;
}

/**
 * The endpoint that receives the traffic from the previous VNF
 * and forwards it to the next SFF / VNF.
 */
app.get('/', async (req: FastifyRequest, res: FastifyReply) => {
    try {
        let sfc: VNF = extractAndValidateSFCHeader(req);
        const sfcID: string = req.headers[ SFC_ID ] as string;

        const sfcTraversedStr: string = req.headers[ SFC_TRAVERSED_HEADER ] as string ?? "";
        let sfcTraversed: VNFUpdated[] = [];
        if (sfcTraversedStr) {
            sfcTraversed = sfcDecode(sfcTraversedStr) as VNF[];
        }

        sfc.isTraversed = true;
        const sfcUpdated: VNFUpdated = { ...sfc };
        delete sfcUpdated.next;
        sfcTraversed.push(sfcUpdated);
        let next: VNF;
        if (Array.isArray(sfc.next)) {
            const network1IP: string = config.sff.network1.networkIP;

            if (checkIPBelongsToNetwork(req.ip ?? "", network1IP)) {
                next = sfc.next[ 0 ] as VNF;
            } else {
                next = sfc.next[ 1 ] as VNF;
            }
        } else {
            next = sfc.next as VNF;
        }

        let nextDest = "";
        if (next.next === TERMINAL) {
            nextDest = `http://${ next.host.ip }`;
        } else if (sfcUpdated.host.id === next.host.id) {
            nextDest = `http://${ next.vnf.ip }`;
        } else {
            nextDest = `http://${ next.host.ip }`;
        }

        next.isTraversed = false;
        const headers = { ...req.headers };
        headers[ SFC_HEADER ] = sfcEncode(next);
        headers[ SFC_TRAVERSED_HEADER ] = sfcEncode(sfcTraversed);

        const requestOptions: AxiosRequestConfig = {
            method: req.method as any,
            url: `${ nextDest }${ req.url.replace("/", "") }`,
            headers,
            data: req.body,
            maxRedirects: 0,
            timeout: config.general.requestTimeout * 1000
        };

        await axios(requestOptions)
            .then((response: AxiosResponse) => {
                res.status(response.status).send(response.data);
            })
            .catch((error: AxiosError) => {
                let msg: string = `[tx] [axios] [${ sfcID }] [${ sfc.vnf.id }] [${ error?.response?.status }] ${ error?.response?.data ?? error.toString() }`;
                logger.error(msg);
                res.status(400).send(msg);
            });
    } catch (error: any) {
        let msg: string = `[tx] [${ req.headers[ SFC_ID ] as string }] ${ error.message ?? error.toString() }`;
        logger.error(msg);
        res.status(400).send(msg);
    }
});

app.listen({ port: config.sff.txPort, host: "0.0.0.0" }, (): void => {
    console.log(`Server is running on port ${ config.sff.txPort }`);
});
