import { isServer, pathUtils } from '@growi/core';
import { Container } from 'unstated';
import urljoin from 'url-join';

import loggerFactory from '~/utils/logger';
import { removeNullPropertyFromObject } from '~/utils/object-utils';

import { apiv3Get, apiv3Put } from '../util/apiv3-client';

const logger = loggerFactory('growi:security:AdminGitHubSecurityContainer');

/**
 * Service container for admin security page (GitHubSecurityManagement.jsx)
 * @extends {Container} unstated Container
 */
export default class AdminGitHubSecurityContainer extends Container {

  constructor(appContainer) {
    super();

    if (isServer()) {
      return;
    }

    this.dummyGithubClientId = 0;
    this.dummyGithubClientIdForError = 1;

    this.state = {
      retrieveError: null,
      // set dummy value tile for using suspense
      githubClientId: this.dummyGithubClientId,
      githubClientSecret: '',
      isSameUsernameTreatedAsIdenticalUser: false,
    };

  }

  /**
   * retrieve security data
   */
  async retrieveSecurityData() {
    try {
      const response = await apiv3Get('/security-setting/');
      const { githubOAuth } = response.data.securityParams;
      this.setState({
        githubClientId: githubOAuth.githubClientId,
        githubClientSecret: githubOAuth.githubClientSecret,
        isSameUsernameTreatedAsIdenticalUser: githubOAuth.isSameUsernameTreatedAsIdenticalUser,
      });
    }
    catch (err) {
      this.setState({ retrieveError: err });
      logger.error(err);
      throw new Error('Failed to fetch data');
    }
  }

  /**
   * Workaround for the mangling in production build to break constructor.name
   */
  static getClassName() {
    return 'AdminGitHubSecurityContainer';
  }

  /**
   * Change githubClientId
   */
  changeGitHubClientId(value) {
    this.setState({ githubClientId: value });
  }

  /**
   * Change githubClientSecret
   */
  changeGitHubClientSecret(value) {
    this.setState({ githubClientSecret: value });
  }

  /**
   * Switch isSameUsernameTreatedAsIdenticalUser
   */
  switchIsSameUsernameTreatedAsIdenticalUser() {
    this.setState({ isSameUsernameTreatedAsIdenticalUser: !this.state.isSameUsernameTreatedAsIdenticalUser });
  }

  /**
   * Update githubSetting
   */
  async updateGitHubSetting() {
    const { githubClientId, githubClientSecret, isSameUsernameTreatedAsIdenticalUser } = this.state;

    let requestParams = { githubClientId, githubClientSecret, isSameUsernameTreatedAsIdenticalUser };

    requestParams = await removeNullPropertyFromObject(requestParams);
    const response = await apiv3Put('/security-setting/github-oauth', requestParams);
    const { securitySettingParams } = response.data;

    this.setState({
      githubClientId: securitySettingParams.githubClientId,
      githubClientSecret: securitySettingParams.githubClientSecret,
      isSameUsernameTreatedAsIdenticalUser: securitySettingParams.isSameUsernameTreatedAsIdenticalUser,
    });
    return response;
  }

}
