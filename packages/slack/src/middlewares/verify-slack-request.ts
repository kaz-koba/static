import { createHmac, timingSafeEqual } from 'crypto';

import { Response, NextFunction } from 'express';
import createError from 'http-errors';
import { stringify } from 'qs';

import { RequestFromSlack } from '../interfaces/request-from-slack';
import loggerFactory from '../utils/logger';

const logger = loggerFactory('@growi/slack:middlewares:verify-slack-request');

/**
 * Verify if the request came from slack
 * See: https://api.slack.com/authentication/verifying-requests-from-slack
 */
export const verifySlackRequest = (req: RequestFromSlack & { rawBody: any }, res: Response, next: NextFunction): Record<string, any> | void => {
  const signingSecret = req.slackSigningSecret;

  if (signingSecret == null) {
    const message = 'No signing secret.';
    logger.warn(message, { body: req.body });
    return next(createError(400, message));
  }

  // take out slackSignature and timestamp from header
  const slackSignature = req.headers['x-slack-signature'];
  const timestamp = req.headers['x-slack-request-timestamp'];

  if (slackSignature == null || timestamp == null) {
    const message = 'Forbidden. Enter from Slack workspace';
    logger.warn(message, { body: req.body });
    return next(createError(403, message));
  }

  // protect against replay attacks
  const time = Math.floor(new Date().getTime() / 1000);
  if (Math.abs(time - timestamp) > 300) {
    const message = 'Verification failed.';
    logger.warn(message, { body: req.body });
    return next(createError(403, message));
  }

  // use req.rawBody for Events API
  // reference: https://stackoverflow.com/questions/64794287/how-to-verify-a-request-from-slack-events-api
  let sigBaseString: string;
  if (req.body.event != null) {
    sigBaseString = `v0:${timestamp}:${req.rawBody}`;
  }
  else {
    sigBaseString = `v0:${timestamp}:${stringify(req.body, { format: 'RFC1738' })}`;
  }
  // generate growi signature
  const hasher = createHmac('sha256', signingSecret);
  hasher.update(sigBaseString, 'utf8');
  const hashedSigningSecret = hasher.digest('hex');
  const growiSignature = `v0=${hashedSigningSecret}`;

  // compare growiSignature and slackSignature
  if (timingSafeEqual(Buffer.from(growiSignature, 'utf8'), Buffer.from(slackSignature, 'utf8'))) {
    return next();
  }

  const message = 'Verification failed.';
  logger.warn(message, { body: req.body });
  return next(createError(403, message));
};
