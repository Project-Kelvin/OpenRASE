import express, { Express, Request, Response } from "express";
import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from "axios";
import { sfcDecode, sfcEncode } from "shared/src/utils/encoder-decoder";
import { getConfig } from "shared/src/utils/config";
import { checkIPBelongsToNetwork } from "shared/src/utils/ip";
import { VNF, VNFUpdated } from "shared/src/models/embedding-graph";
import { SFC_HEADER, SFC_TRAVERSED_HEADER } from "shared/src/constants/sfc";
import { Config } from "shared/src/models/config";
import {TERMINAL} from "shared/src/constants/embedding-graph";

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

        if (sfc.isTraversed) {
            return res.status(400).send(`VNF ${sfc.vnf.id} has already processed this request.`);
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
            .then((response) => {
                res.status(response.status).send(response.data);
            })
            .catch((error: AxiosError) => {
                res.status(400).send(error.response?.data);
            });
    } catch (error: any) {
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

        axios(requestOptions)
            .then((response: AxiosResponse) => {
                res.status(response.status).send(response.data);
            })
            .catch((error: AxiosError) => {
                res.status(400).send(error.response?.data);
            });
    } catch (error: any) {
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
