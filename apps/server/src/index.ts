import fastify, { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { Config } from "shared/models";
import { getConfig } from "shared/utils";

const config: Config = getConfig();

const app: FastifyInstance = fastify({
    ignoreTrailingSlash: true
});
const port: number = config?.server?.port ?? 80;

app.get("/", async (_req: FastifyRequest, res: FastifyReply): Promise<void> => {
    res.send("Hello World!\n");
});

app.listen({ port, host: "0.0.0.0" }, (): void => {
    console.log(`Server is running on port ${port}.`);
});
