import fastify, { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import { getConfig, logger } from "shared/utils";
import { Config } from "shared/models";
import { SFC_ID } from "shared/constants";

const app: FastifyInstance = fastify({
    ignoreTrailingSlash: true
});
const config: Config = getConfig();

app.get('/', async (req: FastifyRequest, res: FastifyReply) => {
    const sffIP: string = config.sff.network1.sffIP;
    const sffPort: number = config.sff.txPort;

    const requestConfig: AxiosRequestConfig = {
        method: req.method as any,
        url: `http://${ sffIP }:${ sffPort }${ req.url }`,
        data: req.body,
        headers: req.headers,
        maxRedirects: 0,
        timeout: config.general.requestTimeout * 1000
    };

    await axios(requestConfig)
        .then((response: AxiosResponse) => {
            res.status(response.status).send(response.data);
        })
        .catch((error: AxiosError) => {
            let msg: string = `[${ req.headers[ SFC_ID ] as string }] ${ error.response?.data ?? error.toString() }`;
            logger.error(msg);
            res.status(error.response?.status ?? 500).send(msg);
        });
});

const port: number = !Number.isNaN(parseInt(process.argv[ 2 ]))
    ? parseInt(process.argv[ 2 ])
    : config?.vnfProxy?.port ?? 80;

app.listen({ port, host: "0.0.0.0" }, (): void => {
    console.log(`Server is running on port ${ port }`);
});
