import { createLogger, format, transports } from "winston";

const myFormat: any = format.printf(({ level, message, timestamp }) => {
    return `${ timestamp } ${ level }: ${ message }`;
});

export const logger = createLogger({
    format: format.combine(
        format.timestamp(),
        myFormat
    ),
    transports: new transports.File({ filename: 'node-logs/error.log', level: 'error' }),
});
