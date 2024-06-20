import express, { Express, Request, Response } from 'express';
import axios, { AxiosError, AxiosResponse } from 'axios';
import { SFC_HEADER, SFC_ID } from "shared/constants";
import { Config, VNF, EmbeddingGraph } from "shared/models";
import { getConfig, sfcEncode, logger } from "shared/utils";
import { IncomingHttpHeaders } from 'http';

const app: Express = express();
const config: Config = getConfig();

const embeddingGraphs: { [ sfcID: string ]: EmbeddingGraph; } = {};


app.use(express.json());

app.post('/add-eg', (req: Request, res: Response) => {
    const eg: EmbeddingGraph = req.body;
    eg.isTraversed = false;

    embeddingGraphs[ eg.sfcID ] = eg;

    res.status(201).send('The Embedding Graph has been successfully added.\n');
});

app.get('/', (req: Request, res: Response) => {
    try {
        if (!req.headers[ SFC_ID ]) {
            logger.error('The SFC-ID Header is missing in the request.');

            return res.status(400).send('The SFC-ID Header is missing in the request.\n');
        }

        const sfcID: string = req.headers[ SFC_ID ] as string;

        if (!embeddingGraphs[ sfcID ]) {
            logger.error(`[${ sfcID }] is not registered with the SFC Classifier.`);

            return res.status(400).send('The SFC-ID is not registered with the SFC Classifier.\n' +
                'Use the `add-eg` endpoint to register it.\n');
        }

        const sfc: VNF = embeddingGraphs[ sfcID ].vnfs;

        const sfcBase64: string = sfcEncode(sfc);

        const headers: IncomingHttpHeaders = { ...req.headers };
        headers[ SFC_HEADER ] = sfcBase64;

        axios.request({
            method: req.method,
            url: `http://${ sfc.host.ip }/rx${ req.url }`,
            data: req.body,
            headers,
            maxRedirects: 0,
            timeout: config.general.requestTimeout * 1000,
        }).then((response: AxiosResponse) => {
            res.status(response.status).send(response.data);
        }).catch((error: AxiosError) => {
            logger.error(`[${ sfcID }] [${error?.response?.status}] ${ error?.response?.data ?? error.toString() }`);
            res.status(400).send(error.response?.data);

        });
    } catch (error: any) {
        logger.error(`[${ req.headers[ SFC_ID ] as string}] ${ error.message ?? error.toString()}`);
        res.status(400).send(error.message);
    }
});

app.listen(config.sfcClassifier.port, '0.0.0.0', () => {
    console.log(`Server is running on port ${ config.sfcClassifier.port }`);
});
