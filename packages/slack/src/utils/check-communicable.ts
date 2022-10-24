
import { WebClient } from '@slack/web-api';
import axios, { AxiosError } from 'axios';

import { ConnectionStatus } from '../interfaces/connection-status';

import { markdownSectionBlock } from './block-kit-builder';
import { requiredScopes } from './required-scopes';
import { generateWebClient } from './webclient-factory';

/**
 * Check whether the HTTP server responds or not.
 *
 * @param serverUri Server URI to connect
 * @returns AxiosError when error is occured
 */
export const connectToHttpServer = async(serverUri: string): Promise<void|AxiosError> => {
  try {
    await axios.get(serverUri, { maxRedirects: 0, timeout: 3000 });
  }
  catch (err) {
    return err as AxiosError;
  }
};

/**
 * Check whether the Slack API server responds or not.
 *
 * @returns AxiosError when error is occured
 */
export const connectToSlackApiServer = async(): Promise<void|AxiosError> => {
  return connectToHttpServer('https://slack.com/api/');
};

/**
 * Test Slack API
 * @param client
 */
const testSlackApiServer = async(client: WebClient): Promise<any> => {
  const result = await client.api.test();

  if (!result.ok) {
    throw new Error(result.error);
  }

  return result;
};

const checkSlackScopes = (resultTestSlackApiServer: any) => {
  const slackScopes = resultTestSlackApiServer.response_metadata.scopes;
  const isPassedScopeCheck = requiredScopes.every(e => slackScopes.includes(e));

  if (!isPassedScopeCheck) {
    throw new Error(`The scopes you registered are not appropriate. Required scopes are ${requiredScopes}`);
  }
};

/**
 * Retrieve Slack workspace name
 * @param client
 */
const retrieveWorkspaceName = async(client: WebClient): Promise<string> => {
  const result = await client.team.info();

  if (!result.ok) {
    throw new Error(result.error);
  }

  return (result as any).team?.name;
};

/**
 * @param token bot OAuth token
 * @returns
 */
export const getConnectionStatus = async(token:string): Promise<ConnectionStatus> => {
  const client = generateWebClient(token);
  const status: ConnectionStatus = {};

  try {
    // try to connect
    const resultTestSlackApiServer = await testSlackApiServer(client);
    // check scope
    await checkSlackScopes(resultTestSlackApiServer);
    // retrieve workspace name
    status.workspaceName = await retrieveWorkspaceName(client);
  }
  catch (err) {
    status.error = err;
  }

  return status;
};

/**
 * Get token string to ConnectionStatus map
 * @param keys Array of bot OAuth token or specific key
 * @param botTokenResolver function to convert from key to token
 * @returns
 */
export const getConnectionStatuses = async(keys: string[], botTokenResolver?: (key: string) => string): Promise<{[key: string]: ConnectionStatus}> => {
  const map = keys
    .reduce<Promise<Map<string, ConnectionStatus>>>(
      async(acc, key) => {
        let token = key;
        if (botTokenResolver != null) {
          token = botTokenResolver(key);
        }
        const status: ConnectionStatus = await getConnectionStatus(token);

        (await acc).set(key, status);
        return acc;
      },
      // define initial accumulator
      Promise.resolve(new Map<string, ConnectionStatus>()),
    );

  // convert to object
  return Object.fromEntries(await map);
};

export const sendSuccessMessage = async(token:string, channel:string, appSiteUrl:string): Promise<void> => {
  const client = generateWebClient(token);
  await client.chat.postMessage({
    channel,
    text: 'Success',
    blocks: [
      markdownSectionBlock(`:tada: Successfully tested with ${appSiteUrl}.`),
      markdownSectionBlock('Now your GROWI and Slack integration is ready to use :+1:'),
    ],
  });
};
