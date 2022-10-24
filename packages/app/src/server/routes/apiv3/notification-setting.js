import { ErrorV3 } from '@growi/core';

import { SupportedAction } from '~/interfaces/activity';
import loggerFactory from '~/utils/logger';
import { removeNullPropertyFromObject } from '~/utils/object-utils';

import { generateAddActivityMiddleware } from '../../middlewares/add-activity';
import { apiV3FormValidator } from '../../middlewares/apiv3-form-validator';
import UpdatePost from '../../models/update-post';


// eslint-disable-next-line no-unused-vars
const logger = loggerFactory('growi:routes:apiv3:notification-setting');

const express = require('express');

const router = express.Router();

const { body } = require('express-validator');

const validator = {
  userNotification: [
    body('pathPattern').isString().trim(),
    body('channel').isString().trim(),
  ],
  globalNotification: [
    body('triggerPath').isString().trim().not()
      .isEmpty(),
    body('notifyToType').isString().trim().isIn(['mail', 'slack']),
    body('toEmail').trim().custom((value, { req }) => {
      return (req.body.notifyToType === 'mail') ? (!!value && value.match(/.+@.+\..+/)) : true;
    }),
    body('slackChannels').trim().custom((value, { req }) => {
      return (req.body.notifyToType === 'slack') ? !!value : true;
    }),
  ],
  notifyForPageGrant: [
    body('isNotificationForOwnerPageEnabled').if(value => value != null).isBoolean(),
    body('isNotificationForGroupPageEnabled').if(value => value != null).isBoolean(),
  ],
};

/**
 * @swagger
 *  tags:
 *    name: NotificationSetting
 */

/**
 * @swagger
 *
 *  components:
 *    schemas:
 *      UserNotificationParams:
 *        type: object
 *        properties:
 *          pathPattern:
 *            type: string
 *            description: path name of wiki
 *          channel:
 *            type: string
 *            description: slack channel name without '#'
 *      NotifyForPageGrant:
 *        type: object
 *        properties:
 *          isNotificationForOwnerPageEnabled:
 *            type: string
 *            description: Whether to notify on owner page
 *          isNotificationForGroupPageEnabled:
 *            type: string
 *            description: Whether to notify on group page
 *      GlobalNotificationParams:
 *        type: object
 *        properties:
 *          notifyToType:
 *            type: string
 *            description: What is type for notify
 *          toEmail:
 *            type: string
 *            description: email for notify
 *          slackChannels:
 *            type: string
 *            description: channels for notify
 *          triggerPath:
 *            type: string
 *            description: trigger path for notify
 *          triggerEvents:
 *            type: array
 *            items:
 *              type: string
 *              description: trigger events for notify
 */
module.exports = (crowi) => {
  const loginRequiredStrictly = require('../../middlewares/login-required')(crowi);
  const adminRequired = require('../../middlewares/admin-required')(crowi);
  const addActivity = generateAddActivityMiddleware(crowi);

  const activityEvent = crowi.event('activity');

  const GlobalNotificationSetting = crowi.model('GlobalNotificationSetting');

  const GlobalNotificationMailSetting = crowi.models.GlobalNotificationMailSetting;
  const GlobalNotificationSlackSetting = crowi.models.GlobalNotificationSlackSetting;

  /**
   * @swagger
   *
   *    /notification-setting/:
   *      get:
   *        tags: [NotificationSetting]
   *        description: Get notification paramators
   *        responses:
   *          200:
   *            description: params of notification
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    notificationParams:
   *                      type: object
   *                      description: notification params
   */
  router.get('/', loginRequiredStrictly, adminRequired, async(req, res) => {

    const notificationParams = {
      // status of slack intagration
      isSlackbotConfigured: crowi.slackIntegrationService.isSlackbotConfigured,
      isSlackLegacyConfigured: crowi.slackIntegrationService.isSlackLegacyConfigured,
      currentBotType: await crowi.configManager.getConfig('crowi', 'slackbot:currentBotType'),

      userNotifications: await UpdatePost.findAll(),
      isNotificationForOwnerPageEnabled: await crowi.configManager.getConfig('notification', 'notification:owner-page:isEnabled'),
      isNotificationForGroupPageEnabled: await crowi.configManager.getConfig('notification', 'notification:group-page:isEnabled'),
      globalNotifications: await GlobalNotificationSetting.findAll(),
    };
    return res.apiv3({ notificationParams });
  });

  /**
  * @swagger
  *
  *    /notification-setting/user-notification:
  *      post:
  *        tags: [NotificationSetting]
  *        description: add user notification setting
  *        requestBody:
  *          required: true
  *          content:
  *            application/json:
  *              schema:
  *                $ref: '#/components/schemas/UserNotificationParams'
  *        responses:
  *          200:
  *            description: Succeeded to add user notification setting
  *            content:
  *              application/json:
  *                schema:
  *                  properties:
  *                    createdUser:
  *                      type: object
  *                      description: user who set notification
  *                    userNotifications:
  *                      type: object
  *                      description: user trigger notifications for updated
  */
  // eslint-disable-next-line max-len
  router.post('/user-notification', loginRequiredStrictly, adminRequired, addActivity, validator.userNotification, apiV3FormValidator, async(req, res) => {
    const { pathPattern, channel } = req.body;

    try {
      logger.info('notification.add', pathPattern, channel);
      const responseParams = {
        createdUser: await UpdatePost.createUpdatePost(pathPattern, channel, req.user),
        userNotifications: await UpdatePost.findAll(),
      };

      const parameters = { action: SupportedAction.ACTION_ADMIN_USER_NOTIFICATION_SETTINGS_ADD };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3({ responseParams }, 201);
    }
    catch (err) {
      const msg = 'Error occurred in updating user notification';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'update-userNotification-failed'));
    }

  });

  /**
   * @swagger
   *
   *    /notification-setting/user-notification/{id}:
   *      delete:
   *        tags: [NotificationSetting]
   *        description: delete user trigger notification pattern
   *        parameters:
   *          - name: id
   *            in: path
   *            required: true
   *            description: id of user trigger notification
   *            schema:
   *              type: string
   *        responses:
   *          200:
   *            description: Succeeded to delete user trigger notification pattern
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    deletedNotificaton:
   *                      type: object
   *                      description: deleted notification
   */
  router.delete('/user-notification/:id', loginRequiredStrictly, adminRequired, addActivity, async(req, res) => {
    const { id } = req.params;

    try {
      const deletedNotificaton = await UpdatePost.findOneAndRemove({ _id: id });

      const parameters = { action: SupportedAction.ACTION_ADMIN_USER_NOTIFICATION_SETTINGS_DELETE };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3(deletedNotificaton);
    }
    catch (err) {
      const msg = 'Error occurred in delete user trigger notification';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'delete-userTriggerNotification-failed'));
    }


  });

  /**
   * @swagger
   *
   *    /notification-setting/global-notification:
   *      post:
   *        tags: [NotificationSetting]
   *        description: add global notification
   *        requestBody:
   *          required: true
   *          content:
   *            application/json:
   *              schema:
   *                $ref: '#/components/schemas/GlobalNotificationParams'
   *        responses:
   *          200:
   *            description: Succeeded to add global notification
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    createdNotification:
   *                      type: object
   *                      description: notification param created
   */
  // eslint-disable-next-line max-len
  router.post('/global-notification', loginRequiredStrictly, adminRequired, addActivity, validator.globalNotification, apiV3FormValidator, async(req, res) => {

    const {
      notifyToType, toEmail, slackChannels, triggerPath, triggerEvents,
    } = req.body;

    let notification;

    if (notifyToType === GlobalNotificationSetting.TYPE.MAIL) {
      notification = new GlobalNotificationMailSetting(crowi);
      notification.toEmail = toEmail;
    }
    if (notifyToType === GlobalNotificationSetting.TYPE.SLACK) {
      notification = new GlobalNotificationSlackSetting(crowi);
      notification.slackChannels = slackChannels;
    }

    notification.triggerPath = triggerPath;
    notification.triggerEvents = triggerEvents || [];

    try {
      const createdNotification = await notification.save();

      const parameters = { action: SupportedAction.ACTION_ADMIN_GLOBAL_NOTIFICATION_SETTINGS_ADD };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3({ createdNotification }, 201);
    }
    catch (err) {
      const msg = 'Error occurred in updating global notification';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'post-globalNotification-failed'));
    }

  });

  /**
   * @swagger
   *
   *    /notification-setting/global-notification/{id}:
   *      put:
   *        tags: [NotificationSetting]
   *        description: update global notification
   *        parameters:
   *          - name: id
   *            in: path
   *            required: true
   *            description: global notification id for updated
   *            schema:
   *              type: string
   *        requestBody:
   *          required: true
   *          content:
   *            application/json:
   *              schema:
   *                $ref: '#/components/schemas/GlobalNotificationParams'
   *        responses:
   *          200:
   *            description: Succeeded to update global notification
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    createdNotification:
   *                      type: object
   *                      description: notification param updated
   */
  // eslint-disable-next-line max-len
  router.put('/global-notification/:id', loginRequiredStrictly, adminRequired, addActivity, validator.globalNotification, apiV3FormValidator, async(req, res) => {
    const { id } = req.params;
    const {
      notifyToType, toEmail, slackChannels, triggerPath, triggerEvents,
    } = req.body;

    const models = {
      [GlobalNotificationSetting.TYPE.MAIL]: GlobalNotificationMailSetting,
      [GlobalNotificationSetting.TYPE.SLACK]: GlobalNotificationSlackSetting,
    };

    try {
      let setting = await GlobalNotificationSetting.findOne({ _id: id });
      setting = setting.toObject();

      // when switching from one type to another,
      // remove toEmail from slack setting and slackChannels from mail setting
      if (setting.__t !== notifyToType) {
        setting = models[setting.__t].hydrate(setting);
        setting.toEmail = undefined;
        setting.slackChannels = undefined;
        await setting.save();
        setting = setting.toObject();
      }

      if (notifyToType === GlobalNotificationSetting.TYPE.MAIL) {
        setting = GlobalNotificationMailSetting.hydrate(setting);
        setting.toEmail = toEmail;
      }
      if (notifyToType === GlobalNotificationSetting.TYPE.SLACK) {
        setting = GlobalNotificationSlackSetting.hydrate(setting);
        setting.slackChannels = slackChannels;
      }

      setting.__t = notifyToType;
      setting.triggerPath = triggerPath;
      setting.triggerEvents = triggerEvents || [];

      const createdNotification = await setting.save();

      const parameters = { action: SupportedAction.ACTION_ADMIN_GLOBAL_NOTIFICATION_SETTINGS_UPDATE };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3({ createdNotification });
    }
    catch (err) {
      const msg = 'Error occurred in updating global notification';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'post-globalNotification-failed'));
    }

  });


  /**
   * @swagger
   *
   *    /notification-setting/notify-for-page-grant:
   *      put:
   *        tags: [NotificationSetting]
   *        description: Update settings for notify for page grant
   *        requestBody:
   *          required: true
   *          content:
   *            application/json:
   *              schema:
   *                $ref: '#/components/schemas/NotifyForPageGrant'
   *        responses:
   *          200:
   *            description: Succeeded to settings for notify for page grant
   *            content:
   *              application/json:
   *                schema:
   *                  $ref: '#/components/schemas/NotifyForPageGrant'
   */
  // eslint-disable-next-line max-len
  router.put('/notify-for-page-grant', loginRequiredStrictly, adminRequired, addActivity, validator.notifyForPageGrant, apiV3FormValidator, async(req, res) => {

    let requestParams = {
      'notification:owner-page:isEnabled': req.body.isNotificationForOwnerPageEnabled,
      'notification:group-page:isEnabled': req.body.isNotificationForGroupPageEnabled,
    };

    requestParams = removeNullPropertyFromObject(requestParams);

    try {
      await crowi.configManager.updateConfigsInTheSameNamespace('notification', requestParams);
      const responseParams = {
        isNotificationForOwnerPageEnabled: await crowi.configManager.getConfig('notification', 'notification:owner-page:isEnabled'),
        isNotificationForGroupPageEnabled: await crowi.configManager.getConfig('notification', 'notification:group-page:isEnabled'),
      };

      const parameters = { action: SupportedAction.ACTION_ADMIN_NOTIFICATION_GRANT_SETTINGS_UPDATE };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3({ responseParams });
    }
    catch (err) {
      const msg = 'Error occurred in updating notify for page grant';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'update-notify-for-page-grant-failed'));
    }

  });

  /**
   * @swagger
   *
   *    /notification-setting/global-notification/{id}/enabled:
   *      put:
   *        tags: [NotificationSetting]
   *        description: toggle enabled global notification
   *        parameters:
   *          - name: id
   *            in: path
   *            required: true
   *            description: notification id for updated
   *            schema:
   *              type: string
   *        requestBody:
   *          required: true
   *          content:
   *            application/json:
   *              schema:
   *                properties:
   *                  isEnabled:
   *                    type: boolean
   *                    description: is notification enabled
   *        responses:
   *          200:
   *            description: Succeeded to delete global notification pattern
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    deletedNotificaton:
   *                      type: object
   *                      description: notification id for updated
   */
  router.put('/global-notification/:id/enabled', loginRequiredStrictly, adminRequired, addActivity, async(req, res) => {
    const { id } = req.params;
    const { isEnabled } = req.body;

    try {
      if (isEnabled) {
        await GlobalNotificationSetting.enable(id);
      }
      else {
        await GlobalNotificationSetting.disable(id);
      }

      const parameters = {
        action: isEnabled
          ? SupportedAction.ACTION_ADMIN_GLOBAL_NOTIFICATION_SETTINGS_ENABLED
          : SupportedAction.ACTION_ADMIN_GLOBAL_NOTIFICATION_SETTINGS_DISABLED,
      };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3({ id });

    }
    catch (err) {
      const msg = 'Error occurred in toggle of global notification';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'toggle-globalNotification-failed'));
    }

  });

  /**
  * @swagger
  *
  *    /notification-setting/global-notification/{id}:
  *      delete:
  *        tags: [NotificationSetting]
  *        description: delete global notification pattern
  *        parameters:
  *          - name: id
  *            in: path
  *            required: true
  *            description: id of global notification
  *            schema:
  *              type: string
  *        responses:
  *          200:
  *            description: Succeeded to delete global notification pattern
  *            content:
  *              application/json:
  *                schema:
  *                  properties:
  *                    deletedNotificaton:
  *                      type: object
  *                      description: deleted notification
  */
  router.delete('/global-notification/:id', loginRequiredStrictly, adminRequired, addActivity, async(req, res) => {
    const { id } = req.params;

    try {
      const deletedNotificaton = await GlobalNotificationSetting.findOneAndRemove({ _id: id });

      const parameters = { action: SupportedAction.ACTION_ADMIN_GLOBAL_NOTIFICATION_SETTINGS_DELETE };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3(deletedNotificaton);
    }
    catch (err) {
      const msg = 'Error occurred in delete global notification';
      logger.error('Error', err);
      return res.apiv3Err(new ErrorV3(msg, 'delete-globalNotification-failed'));
    }

  });

  return router;
};
