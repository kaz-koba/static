import { ErrorV3 } from '@growi/core';

import { SupportedAction } from '~/interfaces/activity';
import loggerFactory from '~/utils/logger';

import { generateAddActivityMiddleware } from '../../middlewares/add-activity';
import { apiV3FormValidator } from '../../middlewares/apiv3-form-validator';

import { SlackbotType, defaultSupportedSlackEventActions } from '@growi/slack';


const {
  getConnectionStatus, getConnectionStatuses,
  sendSuccessMessage,
  defaultSupportedCommandsNameForBroadcastUse, defaultSupportedCommandsNameForSingleUse,
  REQUEST_TIMEOUT_FOR_GTOP,
} = require('@growi/slack');
const axios = require('axios');
const express = require('express');
const { body, query, param } = require('express-validator');
const mongoose = require('mongoose');
const urljoin = require('url-join');

const logger = loggerFactory('growi:routes:apiv3:slack-integration-settings');

const router = express.Router();


/**
 * @swagger
 *  tags:
 *    name: SlackIntegrationSettings
 */

/**
 * @swagger
 *
 *  components:
 *    schemas:
 *      BotType:
 *        description: BotType
 *        properties:
 *          currentBotType:
 *            type: string
 *      SlackIntegration:
 *        description: SlackIntegration
 *        type: object
 *        properties:
 *          currentBotType:
 *            type: string
 */


module.exports = (crowi) => {
  const accessTokenParser = require('../../middlewares/access-token-parser')(crowi);
  const loginRequiredStrictly = require('../../middlewares/login-required')(crowi);
  const adminRequired = require('../../middlewares/admin-required')(crowi);
  const addActivity = generateAddActivityMiddleware(crowi);

  const SlackAppIntegration = crowi.model('SlackAppIntegration');

  const activityEvent = crowi.event('activity');

  const validator = {
    botType: [
      body('currentBotType').isString(),
    ],
    slackIntegration: [
      body('currentBotType')
        .isIn(Object.values(SlackbotType)),
    ],
    proxyUri: [
      body('proxyUri').if(value => value !== '').trim().matches(/^(https?:\/\/)/)
        .isURL({ require_tld: false }),
    ],
    makePrimary: [
      param('id').isMongoId().withMessage('id is required'),
    ],
    updatePermissionsWithoutProxy: [
      body('commandPermission').exists(),
      body('eventActionsPermission').exists(),
      param('id').isMongoId().withMessage('id is required'),
    ],
    updatePermissionsWithProxy: [
      body('permissionsForBroadcastUseCommands').exists(),
      body('permissionsForSingleUseCommands').exists(),
      body('permissionsForSlackEventActions').exists(),
      param('id').isMongoId().withMessage('id is required'),
    ],
    relationTest: [
      param('id').isMongoId(),
      body('channel').trim().isString(),
    ],
    regenerateTokens: [
      param('id').isMongoId(),
    ],
    deleteIntegration: [
      param('id').isMongoId(),
    ],
    slackChannel: [
      body('channel').trim().not().isEmpty()
        .isString(),
    ],
  };

  async function updateSlackBotSettings(params) {
    const { configManager } = crowi;
    // update config without publishing S2sMessage
    return configManager.updateConfigsInTheSameNamespace('crowi', params, true);
  }

  async function resetAllBotSettings(initializedType) {
    await SlackAppIntegration.deleteMany();

    const params = {
      'slackbot:currentBotType': initializedType,
      'slackbot:withoutProxy:signingSecret': null,
      'slackbot:withoutProxy:botToken': null,
      'slackbot:proxyUri': null,
      'slackbot:withoutProxy:commandPermission': null,
      'slackbot:withoutProxy:eventActionsPermission': null,
    };

    return updateSlackBotSettings(params);
  }

  async function getConnectionStatusesFromProxy(tokens) {
    const csv = tokens.join(',');
    const proxyUri = crowi.slackIntegrationService.proxyUriForCurrentType;

    const result = await axios.get(urljoin(proxyUri, '/g2s/connection-status'), {
      headers: {
        'x-growi-gtop-tokens': csv,
        timeout: REQUEST_TIMEOUT_FOR_GTOP,
      },
    });

    return result.data;
  }

  async function requestToProxyServer(token, method, endpoint, body) {
    const proxyUri = crowi.slackIntegrationService.proxyUriForCurrentType;
    if (proxyUri == null) {
      throw new Error('Proxy URL is not registered');
    }

    try {
      const result = await axios[method](
        urljoin(proxyUri, endpoint),
        body, {
          headers: {
            'x-growi-gtop-tokens': token,
          },
          timeout: REQUEST_TIMEOUT_FOR_GTOP,
        },
      );

      return result.data;
    }
    catch (err) {
      throw new Error(`Requesting to proxy server failed: ${err.message}`);
    }
  }

  /**
   * @swagger
   *
   *    /slack-integration-settings/:
   *      get:
   *        tags: [SlackBotSettingParams]
   *        operationId: getSlackBotSettingParams
   *        summary: get /slack-integration
   *        description: Get current settings and connection statuses.
   *        responses:
   *          200:
   *            description: Succeeded to get info.
   */
  router.get('/', accessTokenParser, loginRequiredStrictly, adminRequired, async(req, res) => {

    const { configManager, slackIntegrationService } = crowi;
    const currentBotType = configManager.getConfig('crowi', 'slackbot:currentBotType');

    // retrieve settings
    const settings = {};
    if (currentBotType === SlackbotType.CUSTOM_WITHOUT_PROXY) {
      settings.slackSigningSecretEnvVars = configManager.getConfigFromEnvVars('crowi', 'slackbot:withoutProxy:signingSecret');
      settings.slackBotTokenEnvVars = configManager.getConfigFromEnvVars('crowi', 'slackbot:withoutProxy:botToken');
      settings.slackSigningSecret = configManager.getConfig('crowi', 'slackbot:withoutProxy:signingSecret');
      settings.slackBotToken = configManager.getConfig('crowi', 'slackbot:withoutProxy:botToken');
      settings.commandPermission = configManager.getConfig('crowi', 'slackbot:withoutProxy:commandPermission');
      settings.eventActionsPermission = configManager.getConfig('crowi', 'slackbot:withoutProxy:eventActionsPermission');
    }
    else {
      settings.proxyServerUri = slackIntegrationService.proxyUriForCurrentType;
    }

    // retrieve connection statuses
    let connectionStatuses = {};
    let errorMsg;
    let errorCode;
    if (currentBotType == null) {
      // no need to do anything
    }
    else if (currentBotType === SlackbotType.CUSTOM_WITHOUT_PROXY) {
      const token = settings.slackBotToken;
      // check the token is not null
      if (token != null) {
        try {
          connectionStatuses = await getConnectionStatuses([token]);
        }
        catch (e) {
          errorMsg = 'Error occured in getting connection statuses';
          errorCode = 'get-connection-failed';
          logger.error(errorMsg, e);
        }
      }
    }
    else {
      try {
        const slackAppIntegrations = await SlackAppIntegration.find();
        settings.slackAppIntegrations = slackAppIntegrations;
      }
      catch (e) {
        errorMsg = 'Error occured in finding SlackAppIntegration entities.';
        errorCode = 'get-slackappintegration-failed';
        logger.error(errorMsg, e);
      }

      const proxyServerUri = settings.proxyServerUri;

      if (proxyServerUri != null && settings.slackAppIntegrations != null && settings.slackAppIntegrations.length > 0) {
        try {
          // key: slackAppIntegration.tokenGtoP, value: slackAppIntegration._id
          const tokenGtoPToSlackAppIntegrationId = {};
          settings.slackAppIntegrations.forEach((slackAppIntegration) => {
            tokenGtoPToSlackAppIntegrationId[slackAppIntegration.tokenGtoP] = slackAppIntegration._id;
          });
          const result = (await getConnectionStatusesFromProxy(Object.keys(tokenGtoPToSlackAppIntegrationId)));
          Object.entries(result.connectionStatuses).forEach(([tokenGtoP, connectionStatus]) => {
            connectionStatuses[tokenGtoPToSlackAppIntegrationId[tokenGtoP]] = connectionStatus;
          });
        }
        catch (e) {
          errorMsg = 'Something went wrong when retrieving information from Proxy Server.';
          errorCode = 'test-connection-failed';
          logger.error(errorMsg, e);
        }
      }
    }

    return res.apiv3({
      currentBotType, settings, connectionStatuses, errorMsg, errorCode,
    });
  });


  const handleBotTypeChanging = async(req, res, initializedBotType) => {
    await resetAllBotSettings(initializedBotType);
    crowi.slackIntegrationService.publishUpdatedMessage();

    if (initializedBotType === 'customBotWithoutProxy') {
      // set without-proxy command permissions at bot type changing
      const commandPermission = {};
      [...defaultSupportedCommandsNameForBroadcastUse, ...defaultSupportedCommandsNameForSingleUse].forEach((commandName) => {
        commandPermission[commandName] = true;
      });

      // default event actions permission value
      const eventActionsPermission = {};
      defaultSupportedSlackEventActions.forEach((action) => {
        eventActionsPermission[action] = false;
      });

      const params = {
        'slackbot:withoutProxy:commandPermission': commandPermission,
        'slackbot:withoutProxy:eventActionsPermission': eventActionsPermission,
      };
      try {
        await updateSlackBotSettings(params);
        crowi.slackIntegrationService.publishUpdatedMessage();
      }
      catch (error) {
        const msg = 'Error occured in updating command permission settigns';
        logger.error('Error', error);
        return res.apiv3Err(new ErrorV3(msg, 'update-CustomBotSetting-failed'), 500);
      }
    }

    // TODO Impl to delete AccessToken both of Proxy and GROWI when botType changes.
    const slackBotTypeParam = { slackBotType: crowi.configManager.getConfig('crowi', 'slackbot:currentBotType') };
    return res.apiv3({ slackBotTypeParam });
  };

  /**
   * @swagger
   *
   *    /slack-integration-settings/bot-type/:
   *      put:
   *        tags: [botType]
   *        operationId: putBotType
   *        summary: /slack-integration/bot-type
   *        description: Put botType setting.
   *        requestBody:
   *          required: true
   *          content:
   *            application/json:
   *              schema:
   *                $ref: '#/components/schemas/BotType'
   *        responses:
   *           200:
   *             description: Succeeded to put botType setting.
   */
  // eslint-disable-next-line max-len
  router.put('/bot-type', accessTokenParser, loginRequiredStrictly, adminRequired, addActivity, validator.botType, apiV3FormValidator, async(req, res) => {
    const { currentBotType } = req.body;

    if (currentBotType == null) {
      return res.apiv3Err(new ErrorV3('The param \'currentBotType\' must be specified.', 'update-CustomBotSetting-failed'), 400);
    }

    try {
      await handleBotTypeChanging(req, res, currentBotType);

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_BOT_TYPE_UPDATE });
    }
    catch (error) {
      const msg = 'Error occured in updating Custom bot setting';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'update-CustomBotSetting-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration/bot-type/:
   *      delete:
   *        tags: [botType]
   *        operationId: deleteBotType
   *        summary: /slack-integration/bot-type
   *        description: Delete botType setting.
   *        requestBody:
   *          content:
   *            application/json:
   *              schema:
   *                $ref: '#/components/schemas/BotType'
   *        responses:
   *           200:
   *             description: Succeeded to delete botType setting.
   */
  router.delete('/bot-type', accessTokenParser, loginRequiredStrictly, adminRequired, addActivity, apiV3FormValidator, async(req, res) => {
    try {
      await handleBotTypeChanging(req, res, null);

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_BOT_TYPE_DELETE });
    }
    catch (error) {
      const msg = 'Error occured in resetting all';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'resetting-all-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/without-proxy/update-settings/:
   *      put:
   *        tags: [UpdateWithoutProxySettings]
   *        operationId: putWithoutProxySettings
   *        summary: update customBotWithoutProxy settings
   *        description: Update customBotWithoutProxy setting.
   *        responses:
   *           200:
   *             description: Succeeded to put CustomBotWithoutProxy setting.
   */
  router.put('/without-proxy/update-settings', loginRequiredStrictly, adminRequired, addActivity, async(req, res) => {
    const currentBotType = crowi.configManager.getConfig('crowi', 'slackbot:currentBotType');
    if (currentBotType !== SlackbotType.CUSTOM_WITHOUT_PROXY) {
      const msg = 'Not CustomBotWithoutProxy';
      return res.apiv3Err(new ErrorV3(msg, 'not-customBotWithoutProxy'), 400);
    }

    const { slackSigningSecret, slackBotToken } = req.body;
    const requestParams = {
      'slackbot:withoutProxy:signingSecret': slackSigningSecret,
      'slackbot:withoutProxy:botToken': slackBotToken,
    };
    try {
      await updateSlackBotSettings(requestParams);
      crowi.slackIntegrationService.publishUpdatedMessage();

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_WITHOUT_PROXY_SETTINGS_UPDATE });

      return res.apiv3();
    }
    catch (error) {
      const msg = 'Error occured in updating Custom bot setting';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'update-CustomBotSetting-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/without-proxy/update-permissions/:
   *      put:
   *        tags: [UpdateWithoutProxyPermissions]
   *        operationId: putWithoutProxyPermissions
   *        summary: update customBotWithoutProxy permissions
   *        description: Update customBotWithoutProxy permissions.
   *        responses:
   *           200:
   *             description: Succeeded to put CustomBotWithoutProxy permissions.
   */
  // eslint-disable-next-line max-len
  router.put('/without-proxy/update-permissions', loginRequiredStrictly, adminRequired, addActivity, validator.updatePermissionsWithoutProxy, async(req, res) => {
    const currentBotType = crowi.configManager.getConfig('crowi', 'slackbot:currentBotType');
    if (currentBotType !== SlackbotType.CUSTOM_WITHOUT_PROXY) {
      const msg = 'Not CustomBotWithoutProxy';
      return res.apiv3Err(new ErrorV3(msg, 'not-customBotWithoutProxy'), 400);
    }

    // TODO: look here 78978
    const { commandPermission, eventActionsPermission } = req.body;
    const params = {
      'slackbot:withoutProxy:commandPermission': commandPermission,
      'slackbot:withoutProxy:eventActionsPermission': eventActionsPermission,
    };
    try {
      await updateSlackBotSettings(params);
      crowi.slackIntegrationService.publishUpdatedMessage();

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_WITHOUT_PROXY_PERMISSION_UPDATE });

      return res.apiv3();
    }
    catch (error) {
      const msg = 'Error occured in updating command permission settigns';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'update-CustomBotSetting-failed'), 500);
    }
  });


  /**
   * @swagger
   *
   *    /slack-integration-settings/slack-app-integrations:
   *      post:
   *        tags: [SlackIntegration]
   *        operationId: putSlackAppIntegrations
   *        summary: /slack-integration
   *        description: Generate SlackAppIntegrations
   *        responses:
   *          200:
   *            description: Succeeded to create slack app integration
   */
  router.post('/slack-app-integrations', loginRequiredStrictly, adminRequired, addActivity, async(req, res) => {
    const SlackAppIntegrationRecordsNum = await SlackAppIntegration.countDocuments();
    if (SlackAppIntegrationRecordsNum >= 10) {
      const msg = 'Not be able to create more than 10 slack workspace integration settings';
      logger.error('Error', msg);
      return res.apiv3Err(new ErrorV3(msg, 'create-slackAppIntegeration-failed'), 500);
    }

    const count = await SlackAppIntegration.count();

    const { tokenGtoP, tokenPtoG } = await SlackAppIntegration.generateUniqueAccessTokens();
    try {
      const initialSupportedCommandsForBroadcastUse = new Map(defaultSupportedCommandsNameForBroadcastUse.map(command => [command, true]));
      const initialSupportedCommandsForSingleUse = new Map(defaultSupportedCommandsNameForSingleUse.map(command => [command, true]));
      const initialPermissionsForSlackEventActions = new Map(defaultSupportedSlackEventActions.map(action => [action, true]));

      const slackAppTokens = await SlackAppIntegration.create({
        tokenGtoP,
        tokenPtoG,
        permissionsForBroadcastUseCommands: initialSupportedCommandsForBroadcastUse,
        permissionsForSingleUseCommands: initialSupportedCommandsForSingleUse,
        permissionsForSlackEvents: initialPermissionsForSlackEventActions,
        isPrimary: count === 0,
      });

      const parameters = { action: SupportedAction.ACTION_ADMIN_SLACK_WORKSPACE_CREATE };
      activityEvent.emit('update', res.locals.activity._id, parameters);

      return res.apiv3(slackAppTokens, 200);
    }
    catch (error) {
      const msg = 'Error occurred during creating slack integration settings procedure';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'creating-slack-integration-settings-procedure-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/slack-app-integrations/:id:
   *      delete:
   *        tags: [SlackIntegration]
   *        operationId: deleteAccessTokens
   *        summary: delete accessTokens
   *        description: Delete accessTokens
   *        responses:
   *          200:
   *            description: Succeeded to delete access tokens for slack
   */
  router.delete('/slack-app-integrations/:id', validator.deleteIntegration, apiV3FormValidator, addActivity, async(req, res) => {
    const { id } = req.params;

    try {
      const response = await SlackAppIntegration.findOneAndDelete({ _id: id });

      // update primary
      const countOfPrimary = await SlackAppIntegration.countDocuments({ isPrimary: true });
      if (countOfPrimary === 0) {
        await SlackAppIntegration.updateOne({}, { isPrimary: true });
      }

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_WORKSPACE_DELETE });

      return res.apiv3({ response });
    }
    catch (error) {
      const msg = 'Error occured in deleting access token for slack app tokens';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'update-slackAppTokens-failed'), 500);
    }
  });

  router.put('/proxy-uri', loginRequiredStrictly, adminRequired, addActivity, validator.proxyUri, apiV3FormValidator, async(req, res) => {
    const { proxyUri } = req.body;

    const requestParams = { 'slackbot:proxyUri': proxyUri };

    try {
      await updateSlackBotSettings(requestParams);
      crowi.slackIntegrationService.publishUpdatedMessage();

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_PROXY_URI_UPDATE });

      return res.apiv3({});
    }
    catch (error) {
      const msg = 'Error occured in updating Custom bot setting';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'delete-SlackAppIntegration-failed'), 500);
    }

  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/slack-app-integrations/:id/makeprimary:
   *      put:
   *        tags: [SlackIntegration]
   *        operationId: makePrimary
   *        summary: /slack-integration
   *        description: Make SlackAppTokens primary
   *        responses:
   *          200:
   *            description: Succeeded to make it primary
   */
  // eslint-disable-next-line max-len
  router.put('/slack-app-integrations/:id/make-primary', loginRequiredStrictly, adminRequired, addActivity, validator.makePrimary, apiV3FormValidator, async(req, res) => {

    const { id } = req.params;

    try {
      await SlackAppIntegration.bulkWrite([
        // unset isPrimary for others
        {
          updateMany: {
            filter: { _id: { $ne: id } },
            update: { $unset: { isPrimary: '' } },
          },
        },
        // set primary
        {
          updateOne: {
            filter: { _id: id },
            update: { isPrimary: true },
          },
        },
      ]);

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_MAKE_APP_PRIMARY });

      return res.apiv3();
    }
    catch (error) {
      const msg = 'Error occurred during making SlackAppIntegration primary';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'making-primary-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/slack-app-integrations/:id/regenerate-tokens:
   *      put:
   *        tags: [SlackIntegration]
   *        operationId: putRegenerateTokens
   *        summary: /slack-integration
   *        description: Regenerate SlackAppTokens
   *        responses:
   *          200:
   *            description: Succeeded to regenerate slack app tokens
   */
  // eslint-disable-next-line max-len
  router.put('/slack-app-integrations/:id/regenerate-tokens', loginRequiredStrictly, adminRequired, addActivity, validator.regenerateTokens, apiV3FormValidator, async(req, res) => {

    const { id } = req.params;

    try {
      const { tokenGtoP, tokenPtoG } = await SlackAppIntegration.generateUniqueAccessTokens();
      const slackAppTokens = await SlackAppIntegration.findByIdAndUpdate(id, { tokenGtoP, tokenPtoG });

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_ACCESS_TOKEN_REGENERATE });

      return res.apiv3(slackAppTokens, 200);
    }
    catch (error) {
      const msg = 'Error occurred during regenerating slack app tokens';
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'regenerating-slackAppTokens-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/slack-app-integrations/:id/permissions:
   *      put:
   *        tags: [SlackIntegration]
   *        operationId: putSupportedCommands
   *        summary: /slack-integration-settings/:id/permissions
   *        description: update supported commands
   *        responses:
   *          200:
   *            description: Succeeded to update supported commands
   */
  // eslint-disable-next-line max-len
  router.put('/slack-app-integrations/:id/permissions', loginRequiredStrictly, adminRequired, addActivity, validator.updatePermissionsWithProxy, apiV3FormValidator, async(req, res) => {
    // TODO: look here 78975
    const { permissionsForBroadcastUseCommands, permissionsForSingleUseCommands, permissionsForSlackEventActions } = req.body;
    const { id } = req.params;

    const updatePermissionsForBroadcastUseCommands = new Map(Object.entries(permissionsForBroadcastUseCommands));
    const updatePermissionsForSingleUseCommands = new Map(Object.entries(permissionsForSingleUseCommands));
    const newPermissionsForSlackEventActions = new Map(Object.entries(permissionsForSlackEventActions));


    try {
      const slackAppIntegration = await SlackAppIntegration.findByIdAndUpdate(
        id,
        {
          permissionsForBroadcastUseCommands: updatePermissionsForBroadcastUseCommands,
          permissionsForSingleUseCommands: updatePermissionsForSingleUseCommands,
          permissionsForSlackEventActions: newPermissionsForSlackEventActions,
        },
        { new: true },
      );

      const proxyUri = crowi.slackIntegrationService.proxyUriForCurrentType;
      if (proxyUri != null) {
        await requestToProxyServer(
          slackAppIntegration.tokenGtoP,
          'put',
          '/g2s/supported-commands',
          {
            permissionsForBroadcastUseCommands: slackAppIntegration.permissionsForBroadcastUseCommands,
            permissionsForSingleUseCommands: slackAppIntegration.permissionsForSingleUseCommands,
          },
        );
      }

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_PERMISSION_UPDATE });

      return res.apiv3({});
    }
    catch (error) {
      const msg = `Error occured in updating settings. Cause: ${error.message}`;
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(msg, 'update-permissions-failed'), 500);
    }
  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/slack-app-integrations/:id/relation-test:
   *      post:
   *        tags: [botType]
   *        operationId: postRelationTest
   *        summary: Test relation
   *        description: Delete botType setting.
   *        responses:
   *           200:
   *             description: Succeeded to delete botType setting.
   */
  // eslint-disable-next-line max-len
  router.post('/slack-app-integrations/:id/relation-test', loginRequiredStrictly, adminRequired, addActivity, validator.relationTest, apiV3FormValidator, async(req, res) => {
    const currentBotType = crowi.configManager.getConfig('crowi', 'slackbot:currentBotType');
    if (currentBotType === SlackbotType.CUSTOM_WITHOUT_PROXY) {
      const msg = 'Not Proxy Type';
      return res.apiv3Err(new ErrorV3(msg, 'not-proxy-type'), 400);
    }

    const proxyUri = crowi.slackIntegrationService.proxyUriForCurrentType;
    if (proxyUri == null) {
      return res.apiv3Err(new ErrorV3('Proxy URL is null.', 'not-proxy-Uri'), 400);
    }

    const { id } = req.params;
    let slackBotToken;
    try {
      const slackAppIntegration = await SlackAppIntegration.findOne({ _id: id });
      if (slackAppIntegration == null) {
        const msg = 'Could not find SlackAppIntegration by id';
        return res.apiv3Err(new ErrorV3(msg, 'find-slackAppIntegration-failed'), 400);
      }

      const result = await requestToProxyServer(
        slackAppIntegration.tokenGtoP,
        'post',
        '/g2s/relation-test',
        {
          permissionsForBroadcastUseCommands: slackAppIntegration.permissionsForBroadcastUseCommands,
          permissionsForSingleUseCommands: slackAppIntegration.permissionsForSingleUseCommands,
        },
      );

      slackBotToken = result.slackBotToken;
      if (slackBotToken == null) {
        const msg = 'Could not find slackBotToken by relation';
        return res.apiv3Err(new ErrorV3(msg, 'find-slackBotToken-failed'), 400);
      }
    }
    catch (error) {
      logger.error('Error', error);
      return res.apiv3Err(new ErrorV3(`Error occured while testing. Cause: ${error.message}`, 'test-failed', error.stack));
    }

    const { channel } = req.body;
    const appSiteURL = crowi.configManager.getConfig('crowi', 'app:siteUrl');
    try {
      await sendSuccessMessage(slackBotToken, channel, appSiteURL);
    }
    catch (error) {
      return res.apiv3Err(new ErrorV3(`Error occured while sending message. Cause: ${error.message}`, 'send-message-failed', error.stack));
    }

    activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_RELATION_TEST });

    return res.apiv3();

  });

  /**
   * @swagger
   *
   *    /slack-integration-settings/without-proxy/test:
   *      post:
   *        tags: [botType]
   *        operationId: postTest
   *        summary: test the connection
   *        description: Test the connection with slack work space.
   *        requestBody:
   *          content:
   *            application/json:
   *              schema:
   *                properties:
   *                  testChannel:
   *                    type: string
   *        responses:
   *           200:
   *             description: Succeeded to connect to slack work space.
   */
  router.post('/without-proxy/test', loginRequiredStrictly, adminRequired, addActivity, validator.slackChannel, apiV3FormValidator, async(req, res) => {
    const currentBotType = crowi.configManager.getConfig('crowi', 'slackbot:currentBotType');
    if (currentBotType !== SlackbotType.CUSTOM_WITHOUT_PROXY) {
      const msg = 'Select Without Proxy Type';
      return res.apiv3Err(new ErrorV3(msg, 'select-not-proxy-type'), 400);
    }

    const slackBotToken = crowi.configManager.getConfig('crowi', 'slackbot:withoutProxy:botToken');
    const status = await getConnectionStatus(slackBotToken);
    if (status.error != null) {
      return res.apiv3Err(new ErrorV3(`Error occured while getting connection. ${status.error}`, 'send-message-failed'));
    }

    const { channel } = req.body;
    const appSiteURL = crowi.configManager.getConfig('crowi', 'app:siteUrl');
    try {
      await sendSuccessMessage(slackBotToken, channel, appSiteURL);
    }
    catch (error) {
      return res.apiv3Err(new ErrorV3(`Error occured while sending message. Cause: ${error.message}`, 'send-message-failed', error.stack));
    }

    activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SLACK_WITHOUT_PROXY_TEST });

    return res.apiv3();
  });

  return router;
};
