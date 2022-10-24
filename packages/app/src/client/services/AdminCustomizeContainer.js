import { isServer } from '@growi/core';
import { Container } from 'unstated';

import loggerFactory from '~/utils/logger';

import { toastError } from '../util/apiNotification';
import { apiv3Get, apiv3Put } from '../util/apiv3-client';

// eslint-disable-next-line no-unused-vars
const logger = loggerFactory('growi:services:AdminCustomizeContainer');

/**
 * Service container for admin customize setting page (Customize.jsx)
 * @extends {Container} unstated Container
 */
export default class AdminCustomizeContainer extends Container {

  constructor() {
    super();

    if (isServer()) {
      return;
    }

    this.state = {
      retrieveError: null,
      isEnabledTimeline: false,
      isEnabledAttachTitleHeader: false,

      pageLimitationS: null,
      pageLimitationM: null,
      pageLimitationL: null,
      pageLimitationXL: null,

      isEnabledStaleNotification: false,
      isAllReplyShown: false,
      isSearchScopeChildrenAsDefault: false,
      currentCustomizeTitle: '',
      currentCustomizeHeader: '',
      currentCustomizeCss: '',
      currentCustomizeScript: '',
    };
    this.switchPageListLimitationS = this.switchPageListLimitationS.bind(this);
    this.switchPageListLimitationM = this.switchPageListLimitationM.bind(this);
    this.switchPageListLimitationL = this.switchPageListLimitationL.bind(this);
    this.switchPageListLimitationXL = this.switchPageListLimitationXL.bind(this);

  }

  /**
   * Workaround for the mangling in production build to break constructor.name
   */
  static getClassName() {
    return 'AdminCustomizeContainer';
  }

  /**
   * retrieve customize data
   */
  async retrieveCustomizeData() {
    try {
      const response = await apiv3Get('/customize-setting/');
      const { customizeParams } = response.data;

      this.setState({
        isEnabledTimeline: customizeParams.isEnabledTimeline,
        isEnabledAttachTitleHeader: customizeParams.isEnabledAttachTitleHeader,
        pageLimitationS: customizeParams.pageLimitationS,
        pageLimitationM: customizeParams.pageLimitationM,
        pageLimitationL: customizeParams.pageLimitationL,
        pageLimitationXL: customizeParams.pageLimitationXL,
        isEnabledStaleNotification: customizeParams.isEnabledStaleNotification,
        isAllReplyShown: customizeParams.isAllReplyShown,
        isSearchScopeChildrenAsDefault: customizeParams.isSearchScopeChildrenAsDefault,
        currentCustomizeTitle: customizeParams.customizeTitle,
        currentCustomizeHeader: customizeParams.customizeHeader,
        currentCustomizeCss: customizeParams.customizeCss,
        currentCustomizeScript: customizeParams.customizeScript,
      });
    }
    catch (err) {
      this.setState({ retrieveError: err });
      logger.error(err);
      throw new Error('Failed to fetch data');
    }
  }


  /**
   * Switch enabledTimeLine
   */
  switchEnableTimeline() {
    this.setState({ isEnabledTimeline:  !this.state.isEnabledTimeline });
  }

  /**
   * Switch enabledAttachTitleHeader
   */
  switchEnabledAttachTitleHeader() {
    this.setState({ isEnabledAttachTitleHeader:  !this.state.isEnabledAttachTitleHeader });
  }


  /**
   * S: Switch pageListLimitationS
   */
  switchPageListLimitationS(value) {
    this.setState({ pageLimitationS: value });
  }

  /**
   * M: Switch pageListLimitationM
   */
  switchPageListLimitationM(value) {
    this.setState({ pageLimitationM: value });
  }

  /**
   * L: Switch pageListLimitationL
   */
  switchPageListLimitationL(value) {
    this.setState({ pageLimitationL: value });
  }

  /**
   * XL: Switch pageListLimitationXL
   */
  switchPageListLimitationXL(value) {
    this.setState({ pageLimitationXL: value });
  }

  /**
   * Switch enabledStaleNotification
   */
  switchEnableStaleNotification() {
    this.setState({ isEnabledStaleNotification:  !this.state.isEnabledStaleNotification });
  }

  /**
   * Switch isAllReplyShown
   */
  switchIsAllReplyShown() {
    this.setState({ isAllReplyShown: !this.state.isAllReplyShown });
  }

  /**
   * Switch isSearchScopeChildrenAsDefault
   */
  switchIsSearchScopeChildrenAsDefault() {
    this.setState({ isSearchScopeChildrenAsDefault: !this.state.isSearchScopeChildrenAsDefault });
  }

  /**
   * Change customize Title
   */
  changeCustomizeTitle(inputValue) {
    this.setState({ currentCustomizeTitle: inputValue });
  }

  /**
   * Change customize Html header
   */
  changeCustomizeHeader(inputValue) {
    this.setState({ currentCustomizeHeader: inputValue });
  }

  /**
   * Change customize css
   */
  changeCustomizeCss(inputValue) {
    this.setState({ currentCustomizeCss: inputValue });
  }

  /**
   * Change customize script
   */
  changeCustomizeScript(inpuValue) {
    this.setState({ currentCustomizeScript: inpuValue });
  }


  /**
   * Update function
   * @memberOf AdminCustomizeContainer
   */
  async updateCustomizeFunction() {
    try {
      const response = await apiv3Put('/customize-setting/function', {
        isEnabledTimeline: this.state.isEnabledTimeline,
        isEnabledAttachTitleHeader: this.state.isEnabledAttachTitleHeader,
        pageLimitationS: this.state.pageLimitationS,
        pageLimitationM: this.state.pageLimitationM,
        pageLimitationL: this.state.pageLimitationL,
        pageLimitationXL: this.state.pageLimitationXL,
        isEnabledStaleNotification: this.state.isEnabledStaleNotification,
        isAllReplyShown: this.state.isAllReplyShown,
        isSearchScopeChildrenAsDefault: this.state.isSearchScopeChildrenAsDefault,
      });
      const { customizedParams } = response.data;
      this.setState({
        isEnabledTimeline: customizedParams.isEnabledTimeline,
        isEnabledAttachTitleHeader: customizedParams.isEnabledAttachTitleHeader,
        pageLimitationS: customizedParams.pageLimitationS,
        pageLimitationM: customizedParams.pageLimitationM,
        pageLimitationL: customizedParams.pageLimitationL,
        pageLimitationXL: customizedParams.pageLimitationXL,
        isEnabledStaleNotification: customizedParams.isEnabledStaleNotification,
        isAllReplyShown: customizedParams.isAllReplyShown,
        isSearchScopeChildrenAsDefault: customizedParams.isSearchScopeChildrenAsDefault,
      });
    }
    catch (err) {
      logger.error(err);
      throw new Error('Failed to update data');
    }
  }

  /**
   * Update customTitle
   * @memberOf AdminCustomizeContainer
   */
  async updateCustomizeTitle() {
    try {
      const response = await apiv3Put('/customize-setting/customize-title', {
        customizeTitle: this.state.currentCustomizeTitle,
      });
      const { customizedParams } = response.data;
      this.setState({
        customizeTitle: customizedParams.customizeTitle,
      });
    }
    catch (err) {
      logger.error(err);
      throw new Error('Failed to update data');
    }
  }

  /**
   * Update customHeader
   * @memberOf AdminCustomizeContainer
   */
  async updateCustomizeHeader() {
    try {
      const response = await apiv3Put('/customize-setting/customize-header', {
        customizeHeader: this.state.currentCustomizeHeader,
      });
      const { customizedParams } = response.data;
      this.setState({
        currentCustomizeHeader: customizedParams.customizeHeader,
      });
    }
    catch (err) {
      logger.error(err);
      throw new Error('Failed to update data');
    }
  }

  /**
   * Update customCss
   * @memberOf AdminCustomizeContainer
   */
  async updateCustomizeCss() {
    try {
      const response = await apiv3Put('/customize-setting/customize-css', {
        customizeCss: this.state.currentCustomizeCss,
      });
      const { customizedParams } = response.data;
      this.setState({
        currentCustomizeCss: customizedParams.customizeCss,
      });
    }
    catch (err) {
      logger.error(err);
      throw new Error('Failed to update data');
    }
  }

  /**
   * Update customize script
   * @memberOf AdminCustomizeContainer
   * @return {string} Customize scripts
   */
  async updateCustomizeScript() {
    try {
      const response = await apiv3Put('/customize-setting/customize-script', {
        customizeScript: this.state.currentCustomizeScript,
      });
      const { customizedParams } = response.data;
      this.setState({
        currentCustomizeScript: customizedParams.customizeScript,
      });
    }
    catch (err) {
      logger.error(err);
      throw new Error('Failed to update data');
    }
  }

}
