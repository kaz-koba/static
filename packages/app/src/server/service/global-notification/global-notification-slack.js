import { pagePathUtils } from '@growi/core';
import loggerFactory from '~/utils/logger';


import {
  prepareSlackMessageForGlobalNotification,
} from '../../util/slack';

const logger = loggerFactory('growi:service:GlobalNotificationSlackService'); // eslint-disable-line no-unused-vars
const urljoin = require('url-join');

const { encodeSpaces } = pagePathUtils;

/**
 * sub service class of GlobalNotificationSetting
 */
class GlobalNotificationSlackService {

  constructor(crowi) {
    this.crowi = crowi;

    this.type = crowi.model('GlobalNotificationSetting').TYPE.SLACK;
    this.event = crowi.model('GlobalNotificationSetting').EVENT;
  }


  /**
   * send slack global notification
   *
   * @memberof GlobalNotificationSlackService
   *
   * @param {string} event
   * @param {string} id
   * @param {string} path
   * @param {User} triggeredBy user who triggered the event
   * @param {{ comment: Comment, oldPath: string }} _ event specific vars
   */
  async fire(event, id, path, triggeredBy, vars) {
    const { appService, slackIntegrationService } = this.crowi;

    const GlobalNotification = this.crowi.model('GlobalNotificationSetting');
    const notifications = await GlobalNotification.findSettingByPathAndEvent(event, path, this.type);

    const messageBody = this.generateMessageBody(event, id, path, triggeredBy, vars);
    const attachmentBody = this.generateAttachmentBody(event, id, path, triggeredBy, vars);

    const appTitle = appService.getAppTitle();

    await Promise.all(notifications.map((notification) => {
      const messageObj = prepareSlackMessageForGlobalNotification(messageBody, attachmentBody, appTitle, notification.slackChannels);
      return slackIntegrationService.postMessage(messageObj);
    }));

  }

  /**
   * generate slack message body
   *
   * @memberof GlobalNotificationSlackService
   *
   * @param {string} event event name triggered
   * @param {string} id page id
   * @param {string} path path triggered the event
   * @param {User} triggeredBy user triggered the event
   * @param {{ comment: Comment, oldPath: string }} _ event specific vars
   *
   * @return  {string} slack message body
   */
  generateMessageBody(event, id, path, triggeredBy, { comment, oldPath }) {
    const siteUrl = this.crowi.appService.getSiteUrl();
    const parmaLink = `<${urljoin(siteUrl, id)}|${path}>`;
    const pathLink = `<${urljoin(siteUrl, encodeSpaces(path))}|${path}>`;
    const username = `<${urljoin(siteUrl, 'user', triggeredBy.username)}|${triggeredBy.username}>`;
    let messageBody;

    switch (event) {
      case this.event.PAGE_CREATE:
        messageBody = `:bell: ${username} created ${parmaLink}`;
        break;
      case this.event.PAGE_EDIT:
        messageBody = `:bell: ${username} edited ${parmaLink}`;
        break;
      case this.event.PAGE_DELETE:
        messageBody = `:bell: ${username} deleted ${pathLink}`;
        break;
      case this.event.PAGE_MOVE:
        // validate for page move
        if (oldPath == null) {
          throw new Error(`invalid vars supplied to GlobalNotificationSlackService.generateOption for event ${event}`);
        }
        // eslint-disable-next-line no-case-declarations
        messageBody = `:bell: ${username} moved ${oldPath} to ${parmaLink}`;
        break;
      case this.event.PAGE_LIKE:
        messageBody = `:bell: ${username} liked ${parmaLink}`;
        break;
      case this.event.COMMENT:
        // validate for comment
        if (comment == null) {
          throw new Error(`invalid vars supplied to GlobalNotificationSlackService.generateOption for event ${event}`);
        }
        messageBody = `:bell: ${username} commented on ${parmaLink}`;
        break;
      default:
        throw new Error(`unknown global notificaiton event: ${event}`);
    }

    return messageBody;
  }

  /**
   * generate slack attachment body
   *
   * @memberof GlobalNotificationSlackService
   *
   * @param {string} event event name triggered
   * @param {string} id page id
   * @param {string} path path triggered the event
   * @param {User} triggeredBy user triggered the event
   * @param {{ comment: Comment, oldPath: string }} _ event specific vars
   *
   * @return  {string} slack attachment body
   */
  generateAttachmentBody(event, id, path, triggeredBy, { comment, oldPath }) {
    const attachmentBody = '';

    // TODO: create attachment
    // attachment body is intended for comment or page diff

    // switch (event) {
    //   case this.event.PAGE_CREATE:
    //     break;
    //   case this.event.PAGE_EDIT:
    //     break;
    //   case this.event.PAGE_DELETE:
    //     break;
    //   case this.event.PAGE_MOVE:
    //     break;
    //   case this.event.PAGE_LIKE:
    //     break;
    //   case this.event.COMMENT:
    //     break;
    //   default:
    //     throw new Error(`unknown global notificaiton event: ${event}`);
    // }

    return attachmentBody;
  }

}

module.exports = GlobalNotificationSlackService;
