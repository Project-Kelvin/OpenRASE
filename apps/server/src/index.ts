import express, { Express, Request, Response } from "express";
import { Config } from "shared/models";
import { getConfig } from "shared/utils";

const config: Config = getConfig();

const app: Express = express();
const port: number = config?.server?.port ?? 80;

app.get("/", (_req: Request, res: Response): void => {
    res.send("Hello World!\n");
});

app.listen(port, (): void => {
    console.log(`Server is running on port ${port}.`)
 })
