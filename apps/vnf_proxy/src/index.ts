import express, { Express, Request, Response } from "express";
import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from "axios";
import { getConfig, logger } from "shared/utils";
import { Config } from "shared/models";
import { SFC_ID } from "shared/constants";

const app: Express = express();
const config: Config = getConfig();
app.use(express.json())

app.get('/', (req: Request, res: Response) => {
    const sffIP: string = config.sff.network1.sffIP;
    const sffPort: number = config.sff.port;

    const requestConfig: AxiosRequestConfig = {
        method: req.method,
        url: `http://${ sffIP }:${ sffPort }/tx${ req.url }`,
        data: req.body,
        headers: req.headers,
        timeout: config.general.requestTimeout * 1000,
        maxRedirects: 0
    };

    axios(requestConfig)
        .then((response: AxiosResponse) => {
            res.status(response.status).send(response.data);
        })
        .catch((error: AxiosError) => {
            logger.error(`[${ req.headers[ SFC_ID ] as string}] ${error.response?.data}`);
            res.status(error.response?.status ?? 500).send(error?.response?.data);
        });
});

const port: number = !Number.isNaN(parseInt(process.argv[ 2 ]))
    ? parseInt(process.argv[ 2 ])
    : config?.vnfProxy?.port ?? 80;

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
