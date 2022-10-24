import { ErrorV3 } from '@growi/core';

import { SupportedAction } from '~/interfaces/activity';
import loggerFactory from '~/utils/logger';

import { generateAddActivityMiddleware } from '../../middlewares/add-activity';
import { apiV3FormValidator } from '../../middlewares/apiv3-form-validator';


const logger = loggerFactory('growi:routes:apiv3:search'); // eslint-disable-line no-unused-vars

const express = require('express');
const { body } = require('express-validator');

const router = express.Router();

const noCache = require('nocache');

/**
 * @swagger
 *  tags:
 *    name: Search
 */
module.exports = (crowi) => {
  const accessTokenParser = require('../../middlewares/access-token-parser')(crowi);
  const loginRequired = require('../../middlewares/login-required')(crowi);
  const adminRequired = require('../../middlewares/admin-required')(crowi);
  const addActivity = generateAddActivityMiddleware(crowi);

  const activityEvent = crowi.event('activity');

  /**
   * @swagger
   *
   *  /search/indices:
   *    get:
   *      tags: [Search]
   *      summary: /search/indices
   *      description: Get current status of indices
   *      responses:
   *        200:
   *          description: Status of indices
   *          content:
   *            application/json:
   *              schema:
   *                properties:
   *                  info:
   *                    type: object
   */
  router.get('/indices', noCache(), accessTokenParser, loginRequired, adminRequired, async(req, res) => {
    const { searchService } = crowi;

    if (!searchService.isConfigured) {
      return res.apiv3Err(new ErrorV3('SearchService is not configured', 'search-service-unconfigured'), 503);
    }

    try {
      const info = await searchService.getInfoForAdmin();
      return res.status(200).send({ info });
    }
    catch (err) {
      return res.apiv3Err(err, 503);
    }
  });

  /**
   * @swagger
   *
   *  /search/connection:
   *    get:
   *      tags: [Search]
   *      summary: /search/connection
   *      description: Reconnect to Elasticsearch
   *      responses:
   *        200:
   *          description: Successfully connected
   */
  router.post('/connection', accessTokenParser, loginRequired, adminRequired, addActivity, async(req, res) => {
    const { searchService } = crowi;

    if (!searchService.isConfigured) {
      return res.apiv3Err(new ErrorV3('SearchService is not configured', 'search-service-unconfigured'));
    }

    try {
      await searchService.reconnectClient();

      activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SEARCH_CONNECTION });

      return res.status(200).send();
    }
    catch (err) {
      return res.apiv3Err(err, 503);
    }
  });

  const validatorForPutIndices = [
    body('operation').isString().isIn(['rebuild', 'normalize']),
  ];

  /**
   * @swagger
   *
   *  /search/indices:
   *    put:
   *      tags: [Search]
   *      summary: /search/indices
   *      description: Operate indices
   *      requestBody:
   *        required: true
   *        content:
   *          application/json:
   *            schema:
   *              properties:
   *                operation:
   *                  type: string
   *                  description: Operation type against to indices >
   *                    * `normalize` - Normalize indices
   *                    * `rebuild` - Rebuild indices
   *                  enum: [normalize, rebuild]
   *      responses:
   *        200:
   *          description: Return 200
   */
  router.put('/indices', accessTokenParser, loginRequired, adminRequired, addActivity, validatorForPutIndices, apiV3FormValidator, async(req, res) => {
    const operation = req.body.operation;

    const { searchService } = crowi;

    if (!searchService.isConfigured) {
      return res.apiv3Err(new ErrorV3('SearchService is not configured', 'search-service-unconfigured'));
    }
    if (!searchService.isReachable) {
      return res.apiv3Err(new ErrorV3('SearchService is not reachable', 'search-service-unreachable'));
    }

    try {
      switch (operation) {
        case 'normalize':
          // wait the processing is terminated
          await searchService.normalizeIndices();

          activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SEARCH_INDICES_NORMALIZE });

          return res.status(200).send({ message: 'Operation is successfully processed.' });
        case 'rebuild':
          // NOT wait the processing is terminated
          searchService.rebuildIndex();

          activityEvent.emit('update', res.locals.activity._id, { action: SupportedAction.ACTION_ADMIN_SEARCH_INDICES_REBUILD });

          return res.status(200).send({ message: 'Operation is successfully requested.' });
        default:
          throw new Error(`Unimplemented operation: ${operation}`);
      }
    }
    catch (err) {
      return res.apiv3Err(err, 503);
    }
  });

  return router;
};
