import { parse } from "yaml";
import { readFileSync } from "fs";
import { Config } from "../models/config";

/**
 * This function reads the configuration file and parses it into a configuration object.
 *
 * @returns {Config} The configuration object.
 */
export const getConfig: () => Config = (): Config => parse(readFileSync("config.yaml", "utf8"));
