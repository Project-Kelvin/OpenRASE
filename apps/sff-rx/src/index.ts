import fastify, { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import { getConfig, sfcDecode, logger } from "shared/utils";
import { Config, VNF } from "shared/models";
import { SFC_HEADER, SFC_ID } from "shared/constants";

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
 * The endpoint that receives the traffic from the previous SFF and forwards it to the next VNF.
 */
app.get('/', async (req: FastifyRequest, res: FastifyReply) => {
    try {
        const sfc: VNF = extractAndValidateSFCHeader(req);
        const sfcID: string = req.headers[ SFC_ID ] as string;
        if (sfc.isTraversed) {
            logger.error(`[${ sfcID }] VNF ${ sfc.vnf.id } has already processed this request.`);

            return res.status(400).send(`VNF ${ sfc.vnf.id } has already processed this request.`);
        }

        const vnfIP = sfc.vnf.ip;

        const requestOptions: AxiosRequestConfig = {
            method: req.method as any,
            url: `http://${ vnfIP }${ req.url.replace("/", "") }`,
            headers: req.headers,
            data: req.body,
            maxRedirects: 0,
            timeout: config.general.requestTimeout * 1000
        };

        await axios(requestOptions)
            .then((response: AxiosResponse) => {
                res.status(response.status).send(response.data);
            })
            .catch((error: AxiosError) => {
                let msg: string = `[rx] [axios] [${ sfcID }] [${ sfc.vnf.id }] [${ error?.response?.status }] ${ error?.response?.data?.toString() ?? error.toString() }`;
                logger.error(msg);
                res.status(400).send(msg);
            });
    } catch (error: any) {
        let msg: string = `[rx] [${ req.headers[ SFC_ID ] as string }] ${ error.message ?? error.toString() }`;
        logger.error(msg);
        res.status(400).send(msg);
    }
});

app.listen({ port: config.sff.port, host: "0.0.0.0" }, (): void => {
    console.log(`Server is running on port ${ config.sff.port }`);
});
