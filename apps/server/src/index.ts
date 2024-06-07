import express, { Express, Request, Response } from "express";
import { Config } from "shared/src/models/config";
import { getConfig } from "shared/src/utils/config";

const config: Config = getConfig();

const app: Express = express();
const port: number = config?.server?.port ?? 80;

app.get("/", (_req: Request, res: Response): void => {
    res.send("Hello World!\n");
});

app.listen(port, (): void => {
    console.log(`Server is running on port ${port}.`)
 })
