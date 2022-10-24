import { pagePathUtils } from '@growi/core';
import { body } from 'express-validator';
import mongoose from 'mongoose';
import urljoin from 'url-join';

import { SupportedTargetModel, SupportedAction } from '~/interfaces/activity';
import Activity from '~/server/models/activity';
import XssOption from '~/services/xss/xssOption';
import loggerFactory from '~/utils/logger';

import { PathAlreadyExistsError } from '../models/errors';
import UpdatePost from '../models/update-post';

const { isCreatablePage, isTopPage, isUsersHomePage } = pagePathUtils;
const { serializePageSecurely } = require('../models/serializers/page-serializer');
const { serializeRevisionSecurely } = require('../models/serializers/revision-serializer');
const { serializeUserSecurely } = require('../models/serializers/user-serializer');

/**
 * @swagger
 *  tags:
 *    name: Pages
 */

/**
 * @swagger
 *
 *  components:
 *    schemas:
 *      Page:
 *        description: Page
 *        type: object
 *        properties:
 *          _id:
 *            type: string
 *            description: page ID
 *            example: 5e07345972560e001761fa63
 *          __v:
 *            type: number
 *            description: DB record version
 *            example: 0
 *          commentCount:
 *            type: number
 *            description: count of comments
 *            example: 3
 *          createdAt:
 *            type: string
 *            description: date created at
 *            example: 2010-01-01T00:00:00.000Z
 *          creator:
 *            $ref: '#/components/schemas/User'
 *          extended:
 *            type: object
 *            description: extend data
 *            example: {}
 *          grant:
 *            type: number
 *            description: grant
 *            example: 1
 *          grantedUsers:
 *            type: array
 *            description: granted users
 *            items:
 *              type: string
 *              description: user ID
 *            example: ["5ae5fccfc5577b0004dbd8ab"]
 *          lastUpdateUser:
 *            $ref: '#/components/schemas/User'
 *          liker:
 *            type: array
 *            description: granted users
 *            items:
 *              type: string
 *              description: user ID
 *            example: []
 *          path:
 *            type: string
 *            description: page path
 *            example: /
 *          revision:
 *            $ref: '#/components/schemas/Revision'
 *          status:
 *            type: string
 *            description: status
 *            enum:
 *              - 'wip'
 *              - 'published'
 *              - 'deleted'
 *              - 'deprecated'
 *            example: published
 *          updatedAt:
 *            type: string
 *            description: date updated at
 *            example: 2010-01-01T00:00:00.000Z
 *
 *      UpdatePost:
 *        description: UpdatePost
 *        type: object
 *        properties:
 *          _id:
 *            type: string
 *            description: update post ID
 *            example: 5e0734e472560e001761fa68
 *          __v:
 *            type: number
 *            description: DB record version
 *            example: 0
 *          pathPattern:
 *            type: string
 *            description: path pattern
 *            example: /test
 *          patternPrefix:
 *            type: string
 *            description: patternPrefix prefix
 *            example: /
 *          patternPrefix2:
 *            type: string
 *            description: path
 *            example: test
 *          channel:
 *            type: string
 *            description: channel
 *            example: general
 *          provider:
 *            type: string
 *            description: provider
 *            enum:
 *              - slack
 *            example: slack
 *          creator:
 *            $ref: '#/components/schemas/User'
 *          createdAt:
 *            type: string
 *            description: date created at
 *            example: 2010-01-01T00:00:00.000Z
 */

/* eslint-disable no-use-before-define */
module.exports = function(crowi, app) {
  const debug = require('debug')('growi:routes:page');
  const logger = loggerFactory('growi:routes:page');
  const swig = require('swig-templates');

  const { pathUtils } = require('@growi/core');

  const Page = crowi.model('Page');
  const User = crowi.model('User');
  const PageTagRelation = crowi.model('PageTagRelation');
  const GlobalNotificationSetting = crowi.model('GlobalNotificationSetting');
  const ShareLink = crowi.model('ShareLink');
  const PageRedirect = mongoose.model('PageRedirect');

  const { PageQueryBuilder } = Page;

  const ApiResponse = require('../util/apiResponse');
  const getToday = require('../util/getToday');

  const { configManager, xssService } = crowi;
  const globalNotificationService = crowi.getGlobalNotificationService();
  const userNotificationService = crowi.getUserNotificationService();

  const activityEvent = crowi.event('activity');

  const Xss = require('~/services/xss/index');
  const initializedConfig = {
    isEnabledXssPrevention: configManager.getConfig('markdown', 'markdown:xss:isEnabledPrevention'),
    tagWhiteList: xssService.getTagWhiteList(),
    attrWhiteList: xssService.getAttrWhiteList(),
  };
  const xssOption = new XssOption(initializedConfig);
  const xss = new Xss(xssOption);


  const actions = {};

  function getPathFromRequest(req) {
    return pathUtils.normalizePath(req.pagePath || req.params[0] || req.params.id || '');
  }

  function generatePager(offset, limit, totalCount) {
    let prev = null;

    if (offset > 0) {
      prev = offset - limit;
      if (prev < 0) {
        prev = 0;
      }
    }

    let next = offset + limit;
    if (totalCount < next) {
      next = null;
    }

    return {
      prev,
      next,
      offset,
    };
  }

  function addRenderVarsForPage(renderVars, page) {
    renderVars.page = page;
    renderVars.revision = page.revision;
    renderVars.pageIdOnHackmd = page.pageIdOnHackmd;
    renderVars.revisionHackmdSynced = page.revisionHackmdSynced;
    renderVars.hasDraftOnHackmd = page.hasDraftOnHackmd;

    if (page.creator != null) {
      renderVars.page.creator = renderVars.page.creator.toObject();
    }
    if (page.revision.author != null) {
      renderVars.revision.author = renderVars.revision.author.toObject();
    }
    if (page.deleteUser != null) {
      renderVars.page.deleteUser = renderVars.page.deleteUser.toObject();
    }
  }

  function addRenderVarsForPresentation(renderVars, page) {
    // sanitize page.revision.body
    if (crowi.configManager.getConfig('markdown', 'markdown:xss:isEnabledPrevention')) {
      const preventXssRevision = xss.process(page.revision.body);
      page.revision.body = preventXssRevision;
    }
    renderVars.page = page;
    renderVars.revision = page.revision;
  }

  async function addRenderVarsForUserPage(renderVars, page) {
    const userData = await User.findUserByUsername(User.getUsernameByPath(page.path));

    if (userData != null) {
      renderVars.pageUser = serializeUserSecurely(userData);
    }
  }

  function addRenderVarsForScope(renderVars, page) {
    renderVars.grant = page.grant;
    renderVars.grantedGroupId = page.grantedGroup ? page.grantedGroup.id : null;
    renderVars.grantedGroupName = page.grantedGroup ? page.grantedGroup.name : null;
  }

  async function addRenderVarsForDescendants(renderVars, path, requestUser, offset, limit, isRegExpEscapedFromPath) {
    const SEENER_THRESHOLD = 10;

    const queryOptions = {
      offset,
      limit,
      includeTrashed: path.startsWith('/trash/'),
      isRegExpEscapedFromPath,
    };
    const result = await Page.findListWithDescendants(path, requestUser, queryOptions);
    if (result.pages.length > limit) {
      result.pages.pop();
    }

    renderVars.viewConfig = {
      seener_threshold: SEENER_THRESHOLD,
    };
    renderVars.pager = generatePager(result.offset, result.limit, result.totalCount);
    renderVars.pages = result.pages;
  }

  async function addRenderVarsForPageTree(renderVars, pathOrId, user) {
    const { targetAndAncestors, rootPage } = await Page.findTargetAndAncestorsByPathOrId(pathOrId, user);

    if (targetAndAncestors.length === 0 && pathOrId.includes('/') && !isTopPage(pathOrId)) {
      throw new Error('Ancestors must have at least one page.');
    }

    renderVars.targetAndAncestors = { targetAndAncestors, rootPage };
  }

  async function addRenderVarsWhenNotFound(renderVars, pathOrId) {
    if (pathOrId == null) {
      return;
    }

    renderVars.notFoundTargetPathOrId = pathOrId;
  }

  async function addRenderVarsWhenEmptyPage(renderVars, isEmpty, pageId) {
    if (!isEmpty) return;
    renderVars.pageId = pageId;
    renderVars.isEmpty = isEmpty;
  }

  function replacePlaceholdersOfTemplate(template, req) {
    if (req.user == null) {
      return '';
    }

    const definitions = {
      pagepath: getPathFromRequest(req),
      username: req.user.name,
      today: getToday(),
    };
    const compiledTemplate = swig.compile(template);

    return compiledTemplate(definitions);
  }

  async function _notFound(req, res) {
    const path = getPathFromRequest(req);
    const pathOrId = req.params.id || path;

    let view;
    let action;
    const renderVars = { path };

    if (!isCreatablePage(path)) {
      view = 'layout-growi/not_creatable';
      action = SupportedAction.ACTION_PAGE_NOT_CREATABLE;
    }
    else if (req.isForbidden) {
      view = 'layout-growi/forbidden';
      action = SupportedAction.ACTION_PAGE_FORBIDDEN;
    }
    else {
      view = 'layout-growi/not_found';
      action = SupportedAction.ACTION_PAGE_NOT_FOUND;

      // retrieve templates
      if (req.user != null) {
        const template = await Page.findTemplate(path);

        if (template.templateBody) {
          const body = replacePlaceholdersOfTemplate(template.templateBody, req);
          const tags = template.templateTags;
          renderVars.template = body;
          renderVars.templateTags = tags;
        }
      }

      // add scope variables by ancestor page
      const ancestor = await Page.findAncestorByPathAndViewer(path, req.user);
      if (ancestor != null) {
        await ancestor.populate('grantedGroup');
        addRenderVarsForScope(renderVars, ancestor);
      }
    }

    const limit = 50;
    const offset = parseInt(req.query.offset) || 0;
    await addRenderVarsForDescendants(renderVars, path, req.user, offset, limit, true);
    await addRenderVarsForPageTree(renderVars, pathOrId, req.user);
    await addRenderVarsWhenNotFound(renderVars, pathOrId);
    await addRenderVarsWhenEmptyPage(renderVars, req.isEmpty, req.pageId);

    const parameters = {
      ip:  req.ip,
      endpoint: req.originalUrl,
      action,
      user: req.user?._id,
      snapshot: {
        username: req.user?.username,
      },
    };
    crowi.activityService.createActivity(parameters);

    return res.render(view, renderVars);
  }

  async function showPageForPresentation(req, res, next) {
    const id = req.params.id;
    const { revisionId } = req.query;

    let page = await Page.findByIdAndViewer(id, req.user, null, true, true);

    if (page == null) {
      next();
    }

    // empty page
    if (page.isEmpty) {
      // redirect to page (path) url
      const url = new URL('https://dummy.origin');
      url.pathname = page.path;
      Object.entries(req.query).forEach(([key, value], i) => {
        url.searchParams.append(key, value);
      });
      return res.safeRedirect(urljoin(url.pathname, url.search));

    }

    const renderVars = {};

    // populate
    page = await page.populateDataToMakePresentation(revisionId);

    if (page != null) {
      addRenderVarsForPresentation(renderVars, page);
    }

    return res.render('page_presentation', renderVars);
  }

  async function showTopPage(req, res, next) {
    const portalPath = req.path;
    const revisionId = req.query.revision;

    const view = 'layout-growi/page_list';
    const renderVars = { path: portalPath };

    let portalPage = await Page.findByPathAndViewer(portalPath, req.user);
    portalPage.initLatestRevisionField(revisionId);

    // add user to seen users
    if (req.user != null) {
      portalPage = await portalPage.seen(req.user);
    }

    // populate
    portalPage = await portalPage.populateDataToShowRevision();

    addRenderVarsForPage(renderVars, portalPage);

    const sharelinksNumber = await ShareLink.countDocuments({ relatedPage: portalPage._id });
    renderVars.sharelinksNumber = sharelinksNumber;

    const limit = 50;
    const offset = parseInt(req.query.offset) || 0;

    await addRenderVarsForDescendants(renderVars, portalPath, req.user, offset, limit);

    await addRenderVarsForPageTree(renderVars, portalPath, req.user);

    const parameters = {
      ip:  req.ip,
      endpoint: req.originalUrl,
      action: SupportedAction.ACTION_PAGE_VIEW,
      user: req.user?._id,
      snapshot: {
        username: req.user?.username,
      },
    };
    crowi.activityService.createActivity(parameters);

    return res.render(view, renderVars);
  }

  async function showPageForGrowiBehavior(req, res, next) {
    const id = req.params.id;
    const revisionId = req.query.revision;

    let page = await Page.findByIdAndViewer(id, req.user, null, true, true);

    if (page == null) {
      // check the page is forbidden or just does not exist.
      req.isForbidden = await Page.count({ _id: id }) > 0;
      return _notFound(req, res);
    }

    // empty page
    if (page.isEmpty) {
      req.pageId = page._id;
      req.pagePath = page.path;
      req.isEmpty = page.isEmpty;
      return _notFound(req, res);
    }

    const { path } = page; // this must exist

    logger.debug('Page is found when processing pageShowForGrowiBehavior', page._id, path);

    const limit = 50;
    const offset = parseInt(req.query.offset) || 0;
    const renderVars = {};

    let view = 'layout-growi/page';

    page.initLatestRevisionField(revisionId);

    // add user to seen users
    if (req.user != null) {
      page = await page.seen(req.user);
    }

    // populate
    page = await page.populateDataToShowRevision();
    addRenderVarsForPage(renderVars, page);
    addRenderVarsForScope(renderVars, page);

    await addRenderVarsForDescendants(renderVars, path, req.user, offset, limit, true);

    const sharelinksNumber = await ShareLink.countDocuments({ relatedPage: page._id });
    renderVars.sharelinksNumber = sharelinksNumber;

    if (isUsersHomePage(path)) {
      // change template
      view = 'layout-growi/user_page';
      await addRenderVarsForUserPage(renderVars, page);
    }

    await addRenderVarsForPageTree(renderVars, path, req.user);

    const parameters = {
      ip:  req.ip,
      endpoint: req.originalUrl,
      action: isUsersHomePage(path) ? SupportedAction.ACTION_PAGE_USER_HOME_VIEW : SupportedAction.ACTION_PAGE_VIEW,
      user: req.user?._id,
      snapshot: {
        username: req.user?.username,
      },
    };
    crowi.activityService.createActivity(parameters);

    return res.render(view, renderVars);
  }

  actions.showTopPage = function(req, res) {
    return showTopPage(req, res);
  };

  /**
   * Redirect to the page without trailing slash
   */
  actions.showPageWithEndOfSlash = function(req, res, next) {
    return res.redirect(pathUtils.removeTrailingSlash(req.path));
  };

  /**
   * switch action
   *   - presentation mode
   *   - by behaviorType
   */
  actions.showPage = async function(req, res, next) {
    // presentation mode
    if (req.query.presentation) {
      return showPageForPresentation(req, res, next);
    }
    // delegate to showPageForGrowiBehavior
    return showPageForGrowiBehavior(req, res, next);
  };

  actions.showSharedPage = async function(req, res, next) {
    const { linkId } = req.params;
    const revisionId = req.query.revision;
    const renderVars = {};

    const parameters = {
      ip:  req.ip,
      endpoint: req.originalUrl,
      user: req.user?._id,
      snapshot: {
        username: req.user?.username,
      },
    };

    const shareLink = await ShareLink.findOne({ _id: linkId }).populate('relatedPage');

    if (shareLink == null || shareLink.relatedPage == null || shareLink.relatedPage.isEmpty) {

      Object.assign(parameters, { action: SupportedAction.ACTION_SHARE_LINK_NOT_FOUND });
      crowi.activityService.createActivity(parameters);

      // page or sharelink are not found (or page is empty: abnormaly)
      return res.render('layout-growi/not_found_shared_page');
    }
    if (crowi.configManager.getConfig('crowi', 'security:disableLinkSharing')) {

      Object.assign(parameters, { action: SupportedAction.ACTION_SHARE_LINK_NOT_FOUND });
      crowi.activityService.createActivity(parameters);

      return res.render('layout-growi/forbidden');
    }

    renderVars.sharelink = shareLink;

    // check if share link is expired
    if (shareLink.isExpired()) {
      Object.assign(parameters, { action: SupportedAction.ACTION_SHARE_LINK_EXPIRED_PAGE_VIEW });
      crowi.activityService.createActivity(parameters);

      // page is not found
      return res.render('layout-growi/expired_shared_page', renderVars);
    }

    let page = shareLink.relatedPage;

    // presentation mode
    if (req.query.presentation) {
      page = await page.populateDataToMakePresentation(revisionId);

      // populate
      addRenderVarsForPage(renderVars, page);
      return res.render('page_presentation', renderVars);
    }

    page.initLatestRevisionField(revisionId);

    // populate
    page = await page.populateDataToShowRevision();
    addRenderVarsForPage(renderVars, page);
    addRenderVarsForScope(renderVars, page);

    Object.assign(parameters, { action: SupportedAction.ACTION_SHARE_LINK_PAGE_VIEW });
    crowi.activityService.createActivity(parameters);

    return res.render('layout-growi/shared_page', renderVars);
  };

  /**
   * switch action by behaviorType
   */
  /* eslint-disable no-else-return */
  actions.trashPageShowWrapper = function(req, res) {
    // Crowi behavior for '/trash/*'
    return actions.deletedPageListShow(req, res);
  };
  /* eslint-enable no-else-return */

  /**
   * switch action by behaviorType
   */
  /* eslint-disable no-else-return */
  actions.deletedPageListShowWrapper = function(req, res) {
    const path = `/trash${getPathFromRequest(req)}`;
    return res.redirect(path);
  };
  /* eslint-enable no-else-return */

  actions.notFound = async function(req, res) {
    return _notFound(req, res);
  };

  actions.deletedPageListShow = async function(req, res) {
    // normalizePath makes '/trash/' -> '/trash'
    const path = pathUtils.normalizePath(`/trash${getPathFromRequest(req)}`);

    const limit = 50;
    const offset = parseInt(req.query.offset) || 0;

    const queryOptions = {
      offset,
      limit,
      includeTrashed: true,
    };

    const renderVars = {
      page: null,
      path,
      pages: [],
    };

    const result = await Page.findListWithDescendants(path, req.user, queryOptions);

    if (result.pages.length > limit) {
      result.pages.pop();
    }

    renderVars.pager = generatePager(result.offset, result.limit, result.totalCount);
    renderVars.pages = result.pages;
    res.render('layout-growi/page_list', renderVars);
  };

  /**
   * redirector
   */
  async function redirector(req, res, next, path) {
    const { redirectFrom } = req.query;

    const includeEmpty = true;
    const builder = new PageQueryBuilder(Page.find({ path }), includeEmpty);

    builder.populateDataToList(User.USER_FIELDS_EXCEPT_CONFIDENTIAL);

    await Page.addConditionToFilteringByViewerForList(builder, req.user, true);
    const pages = await builder.query.lean().clone().exec('find');
    const nonEmptyPages = pages.filter(p => !p.isEmpty);

    if (nonEmptyPages.length >= 2) {
      return res.render('layout-growi/identical-path-page', {
        identicalPathPages: nonEmptyPages,
        redirectFrom,
        path,
      });
    }

    if (nonEmptyPages.length === 1) {
      const nonEmptyPage = nonEmptyPages[0];
      const url = new URL('https://dummy.origin');

      url.pathname = `/${nonEmptyPage._id}`;
      Object.entries(req.query).forEach(([key, value], i) => {
        url.searchParams.append(key, value);
      });
      return res.safeRedirect(urljoin(url.pathname, url.search));
    }

    // Processing of nonEmptyPage is finished by the time this code is read
    // If any pages exist then they should be empty
    const emptyPage = pages[0];
    if (emptyPage != null) {
      req.pageId = emptyPage._id;
      req.isEmpty = emptyPage.isEmpty;
      return _notFound(req, res);
    }
    // redirect by PageRedirect
    const pageRedirect = await PageRedirect.findOne({ fromPath: path });
    if (pageRedirect != null) {
      return res.safeRedirect(`${encodeURI(pageRedirect.toPath)}?redirectFrom=${encodeURIComponent(path)}`);
    }

    return _notFound(req, res);
  }

  actions.redirector = async function(req, res, next) {
    const path = getPathFromRequest(req);

    const parameters = {
      ip:  req.ip,
      endpoint: req.originalUrl,
      action: SupportedAction.ACTION_PAGE_VIEW,
      user: req.user?._id,
      snapshot: {
        username: req.user?.username,
      },
    };
    crowi.activityService.createActivity(parameters);
    return redirector(req, res, next, path);
  };

  actions.redirectorWithEndOfSlash = async function(req, res, next) {
    const _path = getPathFromRequest(req);
    const path = pathUtils.removeTrailingSlash(_path);

    const parameters = {
      ip:  req.ip,
      endpoint: req.originalUrl,
      action: SupportedAction.ACTION_PAGE_VIEW,
      user: req.user?._id,
      snapshot: {
        username: req.user?.username,
      },
    };
    crowi.activityService.createActivity(parameters);

    return redirector(req, res, next, path);
  };


  const api = {};
  const validator = {};

  actions.api = api;
  actions.validator = validator;

  /**
   * @swagger
   *
   *    /pages.list:
   *      get:
   *        tags: [Pages, CrowiCompatibles]
   *        operationId: listPages
   *        summary: /pages.list
   *        description: Get list of pages
   *        parameters:
   *          - in: query
   *            name: path
   *            schema:
   *              $ref: '#/components/schemas/Page/properties/path'
   *          - in: query
   *            name: user
   *            schema:
   *              $ref: '#/components/schemas/User/properties/username'
   *          - in: query
   *            name: limit
   *            schema:
   *              $ref: '#/components/schemas/V1PaginateResult/properties/meta/properties/limit'
   *          - in: query
   *            name: offset
   *            schema:
   *              $ref: '#/components/schemas/V1PaginateResult/properties/meta/properties/offset'
   *        responses:
   *          200:
   *            description: Succeeded to get list of pages.
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    ok:
   *                      $ref: '#/components/schemas/V1Response/properties/ok'
   *                    pages:
   *                      type: array
   *                      items:
   *                        $ref: '#/components/schemas/Page'
   *                      description: page list
   *          403:
   *            $ref: '#/components/responses/403'
   *          500:
   *            $ref: '#/components/responses/500'
   */
  /**
   * @api {get} /pages.list List pages by user
   * @apiName ListPage
   * @apiGroup Page
   *
   * @apiParam {String} path
   * @apiParam {String} user
   */
  api.list = async function(req, res) {
    const username = req.query.user || null;
    const path = req.query.path || null;
    const limit = +req.query.limit || 50;
    const offset = parseInt(req.query.offset) || 0;

    const queryOptions = { offset, limit: limit + 1 };

    // Accepts only one of these
    if (username === null && path === null) {
      return res.json(ApiResponse.error('Parameter user or path is required.'));
    }
    if (username !== null && path !== null) {
      return res.json(ApiResponse.error('Parameter user or path is required.'));
    }

    try {
      let result = null;
      if (path == null) {
        const user = await User.findUserByUsername(username);
        if (user === null) {
          throw new Error('The user not found.');
        }
        result = await Page.findListByCreator(user, req.user, queryOptions);
      }
      else {
        result = await Page.findListByStartWith(path, req.user, queryOptions);
      }

      if (result.pages.length > limit) {
        result.pages.pop();
      }

      result.pages.forEach((page) => {
        if (page.lastUpdateUser != null && page.lastUpdateUser instanceof User) {
          page.lastUpdateUser = serializeUserSecurely(page.lastUpdateUser);
        }
      });

      return res.json(ApiResponse.success(result));
    }
    catch (err) {
      return res.json(ApiResponse.error(err));
    }
  };

  // TODO If everything that depends on this route, delete it too
  api.create = async function(req, res) {
    const body = req.body.body || null;
    let pagePath = req.body.path || null;
    const grant = req.body.grant || null;
    const grantUserGroupId = req.body.grantUserGroupId || null;
    const overwriteScopesOfDescendants = req.body.overwriteScopesOfDescendants || null;
    const isSlackEnabled = !!req.body.isSlackEnabled; // cast to boolean
    const slackChannels = req.body.slackChannels || null;
    const pageTags = req.body.pageTags || undefined;

    if (body === null || pagePath === null) {
      return res.json(ApiResponse.error('Parameters body and path are required.'));
    }

    // check whether path starts slash
    pagePath = pathUtils.addHeadingSlash(pagePath);

    // check page existence
    const isExist = await Page.count({ path: pagePath }) > 0;
    if (isExist) {
      return res.json(ApiResponse.error('Page exists', 'already_exists'));
    }

    const options = {};
    if (grant != null) {
      options.grant = grant;
      options.grantUserGroupId = grantUserGroupId;
    }

    const createdPage = await crowi.pageService.create(pagePath, body, req.user, options);

    let savedTags;
    if (pageTags != null) {
      await PageTagRelation.updatePageTags(createdPage.id, pageTags);
      savedTags = await PageTagRelation.listTagNamesByPage(createdPage.id);
    }

    const result = {
      page: serializePageSecurely(createdPage),
      revision: serializeRevisionSecurely(createdPage.revision),
      tags: savedTags,
    };
    res.json(ApiResponse.success(result));

    // update scopes for descendants
    if (overwriteScopesOfDescendants) {
      Page.applyScopesToDescendantsAsyncronously(createdPage, req.user);
    }

    // global notification
    try {
      await globalNotificationService.fire(GlobalNotificationSetting.EVENT.PAGE_CREATE, createdPage, req.user);
    }
    catch (err) {
      logger.error('Create notification failed', err);
    }

    // user notification
    if (isSlackEnabled) {
      try {
        const results = await userNotificationService.fire(createdPage, req.user, slackChannels, 'create');
        results.forEach((result) => {
          if (result.status === 'rejected') {
            logger.error('Create user notification failed', result.reason);
          }
        });
      }
      catch (err) {
        logger.error('Create user notification failed', err);
      }
    }
  };

  /**
   * @swagger
   *
   *    /pages.update:
   *      post:
   *        tags: [Pages, CrowiCompatibles]
   *        operationId: updatePage
   *        summary: /pages.update
   *        description: Update page
   *        requestBody:
   *          content:
   *            application/json:
   *              schema:
   *                properties:
   *                  body:
   *                    $ref: '#/components/schemas/Revision/properties/body'
   *                  page_id:
   *                    $ref: '#/components/schemas/Page/properties/_id'
   *                  revision_id:
   *                    $ref: '#/components/schemas/Revision/properties/_id'
   *                  grant:
   *                    $ref: '#/components/schemas/Page/properties/grant'
   *                required:
   *                  - body
   *                  - page_id
   *                  - revision_id
   *        responses:
   *          200:
   *            description: Succeeded to update page.
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    ok:
   *                      $ref: '#/components/schemas/V1Response/properties/ok'
   *                    page:
   *                      $ref: '#/components/schemas/Page'
   *                    revision:
   *                      $ref: '#/components/schemas/Revision'
   *          403:
   *            $ref: '#/components/responses/403'
   *          500:
   *            $ref: '#/components/responses/500'
   */
  /**
   * @api {post} /pages.update Update page
   * @apiName UpdatePage
   * @apiGroup Page
   *
   * @apiParam {String} body
   * @apiParam {String} page_id
   * @apiParam {String} revision_id
   * @apiParam {String} grant
   *
   * In the case of the page exists:
   * - If revision_id is specified => update the page,
   * - If revision_id is not specified => force update by the new contents.
   */
  api.update = async function(req, res) {
    const pageBody = req.body.body ?? null;
    const pageId = req.body.page_id || null;
    const revisionId = req.body.revision_id || null;
    const grant = req.body.grant || null;
    const grantUserGroupId = req.body.grantUserGroupId || null;
    const overwriteScopesOfDescendants = req.body.overwriteScopesOfDescendants || null;
    const isSlackEnabled = !!req.body.isSlackEnabled; // cast to boolean
    const slackChannels = req.body.slackChannels || null;
    const isSyncRevisionToHackmd = !!req.body.isSyncRevisionToHackmd; // cast to boolean
    const pageTags = req.body.pageTags || undefined;

    if (pageId === null || pageBody === null || revisionId === null) {
      return res.json(ApiResponse.error('page_id, body and revision_id are required.'));
    }

    // check page existence
    const isExist = await Page.count({ _id: pageId }) > 0;
    if (!isExist) {
      return res.json(ApiResponse.error(`Page('${pageId}' is not found or forbidden`, 'notfound_or_forbidden'));
    }

    // check revision
    const Revision = crowi.model('Revision');
    let page = await Page.findByIdAndViewer(pageId, req.user);
    if (page != null && revisionId != null && !page.isUpdatable(revisionId)) {
      const latestRevision = await Revision.findById(page.revision).populate('author');
      const returnLatestRevision = {
        revisionId: latestRevision._id.toString(),
        revisionBody: xss.process(latestRevision.body),
        createdAt: latestRevision.createdAt,
        user: serializeUserSecurely(latestRevision.author),
      };
      return res.json(ApiResponse.error('Posted param "revisionId" is outdated.', 'conflict', returnLatestRevision));
    }

    const options = { isSyncRevisionToHackmd };
    if (grant != null) {
      options.grant = grant;
      options.grantUserGroupId = grantUserGroupId;
    }

    const previousRevision = await Revision.findById(revisionId);
    try {
      page = await Page.updatePage(page, pageBody, previousRevision.body, req.user, options);
    }
    catch (err) {
      logger.error('error on _api/pages.update', err);
      return res.json(ApiResponse.error(err));
    }

    let savedTags;
    if (pageTags != null) {
      const tagEvent = crowi.event('tag');
      await PageTagRelation.updatePageTags(pageId, pageTags);
      savedTags = await PageTagRelation.listTagNamesByPage(pageId);
      tagEvent.emit('update', page, savedTags);
    }

    const result = {
      page: serializePageSecurely(page),
      revision: serializeRevisionSecurely(page.revision),
      tags: savedTags,
    };
    res.json(ApiResponse.success(result));

    // update scopes for descendants
    if (overwriteScopesOfDescendants) {
      Page.applyScopesToDescendantsAsyncronously(page, req.user);
    }

    // global notification
    try {
      await globalNotificationService.fire(GlobalNotificationSetting.EVENT.PAGE_EDIT, page, req.user);
    }
    catch (err) {
      logger.error('Edit notification failed', err);
    }

    // user notification
    if (isSlackEnabled) {
      try {
        const results = await userNotificationService.fire(page, req.user, slackChannels, 'update', { previousRevision });
        results.forEach((result) => {
          if (result.status === 'rejected') {
            logger.error('Create user notification failed', result.reason);
          }
        });
      }
      catch (err) {
        logger.error('Create user notification failed', err);
      }
    }

    const parameters = {
      targetModel: SupportedTargetModel.MODEL_PAGE,
      target: page,
      action: SupportedAction.ACTION_PAGE_UPDATE,
    };
    activityEvent.emit('update', res.locals.activity._id, parameters, { path: page.path, creator: page.creator._id.toString() });
  };

  /**
   * @swagger
   *
   *    /pages.exist:
   *      get:
   *        tags: [Pages]
   *        operationId: getPageExistence
   *        summary: /pages.exist
   *        description: Get page existence
   *        parameters:
   *          - in: query
   *            name: pagePaths
   *            schema:
   *              type: string
   *              description: Page path list in JSON Array format
   *              example: '["/", "/user/unknown"]'
   *        responses:
   *          200:
   *            description: Succeeded to get page existence.
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    ok:
   *                      $ref: '#/components/schemas/V1Response/properties/ok'
   *                    pages:
   *                      type: string
   *                      description: Properties of page path and existence
   *                      example: '{"/": true, "/user/unknown": false}'
   *          403:
   *            $ref: '#/components/responses/403'
   *          500:
   *            $ref: '#/components/responses/500'
   */
  /**
   * @api {get} /pages.exist Get if page exists
   * @apiName GetPage
   * @apiGroup Page
   *
   * @apiParam {String} pages (stringified JSON)
   */
  api.exist = async function(req, res) {
    const pagePaths = JSON.parse(req.query.pagePaths || '[]');

    const pages = {};
    await Promise.all(pagePaths.map(async(path) => {
      // check page existence
      const isExist = await Page.count({ path }) > 0;
      pages[path] = isExist;
      return;
    }));

    const result = { pages };

    return res.json(ApiResponse.success(result));
  };

  /**
   * @swagger
   *
   *    /pages.getPageTag:
   *      get:
   *        tags: [Pages]
   *        operationId: getPageTag
   *        summary: /pages.getPageTag
   *        description: Get page tag
   *        parameters:
   *          - in: query
   *            name: pageId
   *            schema:
   *              $ref: '#/components/schemas/Page/properties/_id'
   *        responses:
   *          200:
   *            description: Succeeded to get page tags.
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    ok:
   *                      $ref: '#/components/schemas/V1Response/properties/ok'
   *                    tags:
   *                      $ref: '#/components/schemas/Tags'
   *          403:
   *            $ref: '#/components/responses/403'
   *          500:
   *            $ref: '#/components/responses/500'
   */
  /**
   * @api {get} /pages.getPageTag get page tags
   * @apiName GetPageTag
   * @apiGroup Page
   *
   * @apiParam {String} pageId
   */
  api.getPageTag = async function(req, res) {
    const result = {};
    try {
      result.tags = await PageTagRelation.listTagNamesByPage(req.query.pageId);
    }
    catch (err) {
      return res.json(ApiResponse.error(err));
    }
    return res.json(ApiResponse.success(result));
  };

  /**
   * @swagger
   *
   *    /pages.updatePost:
   *      get:
   *        tags: [Pages, CrowiCompatibles]
   *        operationId: getUpdatePostPage
   *        summary: /pages.updatePost
   *        description: Get UpdatePost setting list
   *        parameters:
   *          - in: query
   *            name: path
   *            schema:
   *              $ref: '#/components/schemas/Page/properties/path'
   *        responses:
   *          200:
   *            description: Succeeded to get UpdatePost setting list.
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    ok:
   *                      $ref: '#/components/schemas/V1Response/properties/ok'
   *                    updatePost:
   *                      $ref: '#/components/schemas/UpdatePost'
   *          403:
   *            $ref: '#/components/responses/403'
   *          500:
   *            $ref: '#/components/responses/500'
   */
  /**
   * @api {get} /pages.updatePost
   * @apiName Get UpdatePost setting list
   * @apiGroup Page
   *
   * @apiParam {String} path
   */
  api.getUpdatePost = function(req, res) {
    const path = req.query.path;

    if (!path) {
      return res.json(ApiResponse.error({}));
    }

    UpdatePost.findSettingsByPath(path)
      .then((data) => {
        // eslint-disable-next-line no-param-reassign
        data = data.map((e) => {
          return e.channel;
        });
        debug('Found updatePost data', data);
        const result = { updatePost: data };
        return res.json(ApiResponse.success(result));
      })
      .catch((err) => {
        debug('Error occured while get setting', err);
        return res.json(ApiResponse.error({}));
      });
  };

  validator.remove = [
    body('completely')
      .custom(v => v === 'true' || v === true || v == null)
      .withMessage('The body property "completely" must be "true" or true. (Omit param for false)'),
    body('recursively')
      .custom(v => v === 'true' || v === true || v == null)
      .withMessage('The body property "recursively" must be "true" or true. (Omit param for false)'),
  ];

  /**
   * @api {post} /pages.remove Remove page
   * @apiName RemovePage
   * @apiGroup Page
   *
   * @apiParam {String} page_id Page Id.
   * @apiParam {String} revision_id
   */
  api.remove = async function(req, res) {
    const pageId = req.body.page_id;
    const previousRevision = req.body.revision_id || null;

    const { recursively: isRecursively, completely: isCompletely } = req.body;

    const options = {};

    const activityParameters = {
      ip: req.ip,
      endpoint: req.originalUrl,
    };

    const page = await Page.findByIdAndViewer(pageId, req.user, null, true);

    if (page == null) {
      return res.json(ApiResponse.error(`Page '${pageId}' is not found or forbidden`, 'notfound_or_forbidden'));
    }

    let creator;
    if (page.isEmpty) {
      // If empty, the creator is inherited from the closest non-empty ancestor page.
      const notEmptyClosestAncestor = await Page.findNonEmptyClosestAncestor(page.path);
      creator = notEmptyClosestAncestor.creator;
    }
    else {
      creator = page.creator;
    }

    debug('Delete page', page._id, page.path);

    try {
      if (isCompletely) {
        if (!crowi.pageService.canDeleteCompletely(page.path, creator, req.user, isRecursively)) {
          return res.json(ApiResponse.error('You can not delete this page completely', 'user_not_admin'));
        }
        await crowi.pageService.deleteCompletely(page, req.user, options, isRecursively, false, activityParameters);
      }
      else {
        // behave like not found
        const notRecursivelyAndEmpty = page.isEmpty && !isRecursively;
        if (notRecursivelyAndEmpty) {
          return res.json(ApiResponse.error(`Page '${pageId}' is not found.`, 'notfound'));
        }

        if (!page.isEmpty && !page.isUpdatable(previousRevision)) {
          return res.json(ApiResponse.error('Someone could update this page, so couldn\'t delete.', 'outdated'));
        }

        if (!crowi.pageService.canDelete(page.path, creator, req.user, isRecursively)) {
          return res.json(ApiResponse.error('You can not delete this page', 'user_not_admin'));
        }

        await crowi.pageService.deletePage(page, req.user, options, isRecursively, activityParameters);
      }
    }
    catch (err) {
      logger.error('Error occured while get setting', err);
      return res.json(ApiResponse.error('Failed to delete page.', err.message));
    }

    debug('Page deleted', page.path);
    const result = {};
    result.path = page.path;
    result.isRecursively = isRecursively;
    result.isCompletely = isCompletely;

    res.json(ApiResponse.success(result));

    try {
      // global notification
      await globalNotificationService.fire(GlobalNotificationSetting.EVENT.PAGE_DELETE, page, req.user);
    }
    catch (err) {
      logger.error('Delete notification failed', err);
    }
  };

  validator.revertRemove = [
    body('recursively')
      .optional()
      .custom(v => v === 'true' || v === true || v == null)
      .withMessage('The body property "recursively" must be "true" or true. (Omit param for false)'),
  ];

  /**
   * @api {post} /pages.revertRemove Revert removed page
   * @apiName RevertRemovePage
   * @apiGroup Page
   *
   * @apiParam {String} page_id Page Id.
   */
  api.revertRemove = async function(req, res, options) {
    const pageId = req.body.page_id;

    // get recursively flag
    const isRecursively = req.body.recursively;

    const activityParameters = {
      ip: req.ip,
      endpoint: req.originalUrl,
    };

    let page;
    let descendantPages;
    try {
      page = await Page.findByIdAndViewer(pageId, req.user);
      if (page == null) {
        throw new Error(`Page '${pageId}' is not found or forbidden`, 'notfound_or_forbidden');
      }
      page = await crowi.pageService.revertDeletedPage(page, req.user, {}, isRecursively, activityParameters);
    }
    catch (err) {
      if (err instanceof PathAlreadyExistsError) {
        logger.error('Path already exists', err);
        return res.json(ApiResponse.error(err, 'already_exists', err.targetPath));
      }
      logger.error('Error occured while get setting', err);
      return res.json(ApiResponse.error(err));
    }

    const result = {};
    result.page = page; // TODO consider to use serializePageSecurely method -- 2018.08.06 Yuki Takei

    return res.json(ApiResponse.success(result));
  };

  /**
   * @swagger
   *
   *    /pages.duplicate:
   *      post:
   *        tags: [Pages]
   *        operationId: duplicatePage
   *        summary: /pages.duplicate
   *        description: Duplicate page
   *        requestBody:
   *          content:
   *            application/json:
   *              schema:
   *                properties:
   *                  page_id:
   *                    $ref: '#/components/schemas/Page/properties/_id'
   *                  new_path:
   *                    $ref: '#/components/schemas/Page/properties/path'
   *                required:
   *                  - page_id
   *        responses:
   *          200:
   *            description: Succeeded to duplicate page.
   *            content:
   *              application/json:
   *                schema:
   *                  properties:
   *                    ok:
   *                      $ref: '#/components/schemas/V1Response/properties/ok'
   *                    page:
   *                      $ref: '#/components/schemas/Page'
   *                    tags:
   *                      $ref: '#/components/schemas/Tags'
   *          403:
   *            $ref: '#/components/responses/403'
   *          500:
   *            $ref: '#/components/responses/500'
   */
  /**
   * @api {post} /pages.duplicate Duplicate page
   * @apiName DuplicatePage
   * @apiGroup Page
   *
   * @apiParam {String} page_id Page Id.
   * @apiParam {String} new_path New path name.
   */
  api.duplicate = async function(req, res) {
    const pageId = req.body.page_id;
    let newPagePath = pathUtils.normalizePath(req.body.new_path);

    const page = await Page.findByIdAndViewer(pageId, req.user);

    if (page == null) {
      return res.json(ApiResponse.error(`Page '${pageId}' is not found or forbidden`, 'notfound_or_forbidden'));
    }

    // check whether path starts slash
    newPagePath = pathUtils.addHeadingSlash(newPagePath);

    await page.populateDataToShowRevision();
    const originTags = await page.findRelatedTagsById();

    req.body.path = newPagePath;
    req.body.body = page.revision.body;
    req.body.grant = page.grant;
    req.body.grantedUsers = page.grantedUsers;
    req.body.grantUserGroupId = page.grantedGroup;
    req.body.pageTags = originTags;

    return api.create(req, res);
  };

  /**
   * @api {post} /pages.unlink Remove the redirecting page
   * @apiName UnlinkPage
   * @apiGroup Page
   *
   * @apiParam {String} page_id Page Id.
   * @apiParam {String} revision_id
   */
  api.unlink = async function(req, res) {
    const path = req.body.path;

    try {
      await PageRedirect.removePageRedirectsByToPath(path);
      logger.debug('Redirect Page deleted', path);
    }
    catch (err) {
      logger.error('Error occured while get setting', err);
      return res.json(ApiResponse.error('Failed to delete redirect page.'));
    }

    const result = { path };
    return res.json(ApiResponse.success(result));
  };

  return actions;
};
