import express, { Express, Request, Response } from "express";
import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from "axios";
import { checkIPBelongsToNetwork, getConfig, sfcDecode, sfcEncode, logger } from "shared/utils";
import { Config, VNF, VNFUpdated } from "shared/models";
import { TERMINAL, SFC_HEADER, SFC_TRAVERSED_HEADER, SFC_ID } from "shared/constants";
import http from "http";

const app: Express = express();
const config: Config = getConfig();

app.use(express.json());

// In-memory list of host IP addresses.
const hosts: string[] = [];

/**
 * Extract and validate the SFC header from the request.
 *
 * @param request The request received by the SFF.
 * @returns The SFC metadata.
 *
 * @throws If the SFC header is not found in the request.
 * @throws If the request is sent to the wrong host.
 */
function extractAndValidateSFCHeader(request: Request): VNF {
    const sfcBase64: string = request.headers[SFC_HEADER] as string ?? "";

    if (hosts.length === 0) {
        throw new Error(
            'This SFF has no hosts assigned to it.\n' +
                'Please add the host IP address using the `/add-host` endpoint.\n'
        );
    }

    if (!sfcBase64) {
        throw new Error(
            `The SFF running on ${hosts.join(', ')} could not find the SFC header ` +
                `attribute in the request from: ${request.ip}.\n`
        );
    }

    const sfc: VNF = sfcDecode(sfcBase64) as VNF;

    if (sfc.host.ip && !hosts.includes(sfc.host.ip)) {
        throw new Error(
            'This request arrived at the wrong host.\n' +
                'This host has the following IP addresses:\n' +
                `${hosts.join(', ')}.\n` +
                `However, this request was sent to ${sfc.host}.\n`
        );
    }

    return sfc;
}

/**
 * The endpoint that receives the traffic from the previous SFF and forwards it to the next VNF.
 */
app.get('/rx', (req: Request, res: Response) => {
    try {
        const sfc: VNF = extractAndValidateSFCHeader(req);
        const sfcID: string = req.headers[SFC_ID] as string;
        if (sfc.isTraversed) {
            logger.error(`[${sfcID}] VNF ${sfc.vnf.id} has already processed this request.`);

            return res.status(400).send(`VNF ${ sfc.vnf.id } has already processed this request.`);
        }

        const vnfIP = sfc.vnf.ip;

        const requestOptions: AxiosRequestConfig = {
            method: req.method,
            url: `http://${vnfIP}${req.url.replace("/rx","")}`,
            headers: req.headers,
            data: req.body,
            maxRedirects: 0,
            timeout: config.general.requestTimeout * 1000,
        };

        axios(requestOptions)
            .then((response: AxiosResponse) => {
                res.status(response.status).send(response.data);
            })
            .catch((error: AxiosError) => {
                logger.error(`[rx] [axios] [${ sfcID }] [${sfc.vnf.id}] [${error?.response?.status}] ${error?.response?.data ?? error.toString()}`);
                res.status(400).send(error.response?.data);
            });
    } catch (error: any) {
        logger.error(`[rx] [${ req.headers[ SFC_ID ] as string}] ${error.message ?? error.toString()}`);
        res.status(400).send(error.message);
    }
});

/**
 * The endpoint that receives the traffic from the previous VNF
 * and forwards it to the next SFF / VNF.
 */
app.get('/tx', (req: Request, res: Response) => {
    try {
        let sfc: VNF = extractAndValidateSFCHeader(req);
        const sfcID: string = req.headers[SFC_ID] as string;

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
                next = sfc.next[0] as VNF;
            } else {
                next = sfc.next[1] as VNF;
            }
        } else {
            next = sfc.next as VNF;
        }

        let nextDest = "";
        if (next.next === TERMINAL) {
            nextDest = `http://${ next.host.ip }`;
        } else if (sfcUpdated.host.id === next.host.id) {
            nextDest = `http://${next.vnf.ip}`;
        } else {
            nextDest = `http://${next.host.ip}/rx`;
        }

        next.isTraversed = false;
        const headers = { ...req.headers };
        headers[ SFC_HEADER ] = sfcEncode(next);
        headers[SFC_TRAVERSED_HEADER] = sfcEncode(sfcTraversed);

        const requestOptions: AxiosRequestConfig = {
            method: req.method,
            url: `${nextDest}${req.url.replace("/tx", "")}`,
            headers,
            data: req.body,
            maxRedirects: 0,
            timeout: config.general.requestTimeout * 1000,
        };

        if (sfcUpdated.host.id !== next.host.id) {
            const options = { localAddress: sfcUpdated.host.ip };
            const httpAgent = new http.Agent(options);
            requestOptions.httpAgent = httpAgent;
        }

        axios(requestOptions)
            .then((response: AxiosResponse) => {
                res.status(response.status).send(response.data);
            })
            .catch((error: AxiosError) => {
                logger.error(`[tx] [axios] [${ sfcID }] [${ sfc.vnf.id }] [${error?.response?.status}] ${ error?.response?.data ?? error.toString()}`);
                res.status(400).send(error.response?.data);
            });
    } catch (error: any) {
        logger.error(`[tx] [${ req.headers[ SFC_ID ] as string}] ${error.message ?? error.toString()}`);
        res.status(400).send(error.message);
    }
});

/**
 * The endpoint that adds the IP address assigned to the host the SFF is running on
 * to an in-memory list.
 */
app.post('/add-host', (req: Request, res: Response) => {
    const ipAddress: string = req.body.hostIP;

    hosts.push(ipAddress);

    res.sendStatus(200);
});

app.listen(config.sff.port, '0.0.0.0', () => {
    console.log(`Server is running on port ${config.sff.port}`);
});
