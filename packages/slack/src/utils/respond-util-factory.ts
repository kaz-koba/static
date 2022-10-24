import axios from 'axios';
import urljoin from 'url-join';

import { IRespondUtil } from '../interfaces/respond-util';
import { RespondBodyForResponseUrl } from '../interfaces/response-url';

type AxiosOptions = {
  headers?: {
    [header:string]: string,
  }
}

function getResponseUrlForProxy(proxyUri: string, responseUrl: string): string {
  return urljoin(proxyUri, `/g2s/respond?response_url=${responseUrl}`);
}

function getUrl(responseUrl: string, proxyUri: string | null): string {
  return proxyUri == null ? responseUrl : getResponseUrlForProxy(proxyUri, responseUrl);
}

export class RespondUtil implements IRespondUtil {

  url!: string;

  options!: AxiosOptions;

  constructor(responseUrl: string, proxyUri: string | null, appSiteUrl: string) {
    this.url = getUrl(responseUrl, proxyUri);

    this.options = {
      headers: {
        'x-growi-app-site-url': appSiteUrl,
      },
    };
  }

  async respond(body: RespondBodyForResponseUrl): Promise<void> {
    return axios.post(this.url, {
      replace_original: false,
      text: body.text,
      blocks: body.blocks,
    }, this.options);
  }

  async respondInChannel(body: RespondBodyForResponseUrl): Promise<void> {
    return axios.post(this.url, {
      response_type: 'in_channel',
      replace_original: false,
      text: body.text,
      blocks: body.blocks,
    }, this.options);
  }

  async replaceOriginal(body: RespondBodyForResponseUrl): Promise<void> {
    return axios.post(this.url, {
      replace_original: true,
      text: body.text,
      blocks: body.blocks,
    }, this.options);
  }

  async deleteOriginal(): Promise<void> {
    return axios.post(this.url, {
      delete_original: true,
    }, this.options);
  }

}

export function generateRespondUtil(responseUrl: string, proxyUri: string | null, appSiteUrl: string): RespondUtil {
  return new RespondUtil(responseUrl, proxyUri, appSiteUrl);
}
