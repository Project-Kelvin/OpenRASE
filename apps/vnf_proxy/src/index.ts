import express, { Express, Request, Response } from "express";
import axios from "axios";
import { getConfig } from "shared/src/utils/config";
import { Config } from "shared/src/models/config";

const app: Express = express();
const config: Config = getConfig();

app.get('/', async (req: Request, res: Response) => {
    const sffIP: string = config.sff.network1.sffIP;
    const sffPort: number = config.sff.port;

    try {
        const response = await axios.request({
            method: req.method,
            url: `http://${ sffIP }:${ sffPort }/tx${ req.url }`,
            data: req.body,
            headers: req.headers,
            timeout: config.general.requestTimeout * 5000,
            maxRedirects: 0
        });

        res.status(response.status).send(response.data);
    } catch (error: any) {
        res.status(500).send(`Internal Server Error. ${error.toString()}`);
    }
});

const port: number = !Number.isNaN(parseInt(process.argv[ 2 ]))
    ? parseInt(process.argv[ 2 ])
    : config?.vnfProxy?.port ?? 80;

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
