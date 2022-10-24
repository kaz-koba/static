import React, { useCallback } from 'react';

import { useTranslation, i18n } from 'next-i18next';
import PropTypes from 'prop-types';

import { i18n as i18nConfig } from '^/config/next-i18next.config';

import AdminAppContainer from '~/client/services/AdminAppContainer';
import { toastSuccess, toastError } from '~/client/util/apiNotification';
import loggerFactory from '~/utils/logger';


import { withUnstatedContainers } from '../../UnstatedUtils';
import AdminUpdateButtonRow from '../Common/AdminUpdateButtonRow';

const logger = loggerFactory('growi:appSettings');


const AppSetting = (props) => {
  const { adminAppContainer } = props;
  const { t } = useTranslation(['admin', 'commons']);

  const submitHandler = useCallback(async() => {
    try {
      await adminAppContainer.updateAppSettingHandler();
      toastSuccess(t('toaster.update_successed', { target: t('commons:headers.app_settings') }));
    }
    catch (err) {
      toastError(err);
      logger.error(err);
    }
  }, [adminAppContainer, t]);


  return (
    <React.Fragment>
      <div className="form-group row">
        <label className="text-left text-md-right col-md-3 col-form-label">{t('admin:app_setting.site_name')}</label>
        <div className="col-md-6">
          <input
            className="form-control"
            type="text"
            defaultValue={adminAppContainer.state.title || ''}
            onChange={(e) => {
              adminAppContainer.changeTitle(e.target.value);
            }}
            placeholder="GROWI"
          />
          <p className="form-text text-muted">{t('admin:app_setting.sitename_change')}</p>
        </div>
      </div>

      <div className="row form-group mb-5">
        <label
          className="text-left text-md-right col-md-3 col-form-label"
        >
          {t('admin:app_setting.confidential_name')}
        </label>
        <div className="col-md-6">
          <input
            className="form-control"
            type="text"
            defaultValue={adminAppContainer.state.confidential || ''}
            onChange={(e) => {
              adminAppContainer.changeConfidential(e.target.value);
            }}
            placeholder={t('admin:app_setting.confidential_example')}
          />
          <p className="form-text text-muted">{t('admin:app_setting.header_content')}</p>
        </div>
      </div>

      <div className="row form-group mb-5">
        <label
          className="text-left text-md-right col-md-3 col-form-label"
        >
          {t('admin:app_setting.default_language')}
        </label>
        <div className="col-md-6 py-2">
          {
            i18nConfig.locales.map((locale) => {
              if (i18n == null) { return }
              const fixedT = i18n.getFixedT(locale, 'admin');

              return (
                <div key={locale} className="custom-control custom-radio custom-control-inline">
                  <input
                    type="radio"
                    id={`radioLang${locale}`}
                    className="custom-control-input"
                    name="globalLang"
                    value={locale}
                    checked={adminAppContainer.state.globalLang === locale}
                    onChange={(e) => {
                      adminAppContainer.changeGlobalLang(e.target.value);
                    }}
                  />
                  <label className="custom-control-label" htmlFor={`radioLang${locale}`}>{fixedT('meta.display_name')}</label>
                </div>
              );
            })
          }
        </div>
      </div>

      <div className="row form-group mb-5">
        <label
          className="text-left text-md-right col-md-3 col-form-label"
        >
          {t('admin:app_setting.default_mail_visibility')}
        </label>
        <div className="col-md-6 py-2">

          <div className="custom-control custom-radio custom-control-inline">
            <input
              type="radio"
              id="radio-email-show"
              className="custom-control-input"
              name="mailVisibility"
              checked={adminAppContainer.state.isEmailPublishedForNewUser === true}
              onChange={() => { adminAppContainer.changeIsEmailPublishedForNewUserShow(true) }}
            />
            <label className="custom-control-label" htmlFor="radio-email-show">{t('Show')}</label>
          </div>

          <div className="custom-control custom-radio custom-control-inline">
            <input
              type="radio"
              id="radio-email-hide"
              className="custom-control-input"
              name="mailVisibility"
              checked={adminAppContainer.state.isEmailPublishedForNewUser === false}
              onChange={() => { adminAppContainer.changeIsEmailPublishedForNewUserShow(false) }}
            />
            <label className="custom-control-label" htmlFor="radio-email-hide">{t('Hide')}</label>
          </div>

        </div>
      </div>

      <div className="row form-group mb-5">
        <label
          className="text-left text-md-right col-md-3 col-form-label"
        >
          {/* {t('admin:app_setting.file_uploading')} */}
        </label>
        <div className="col-md-6">
          <div className="custom-control custom-checkbox custom-checkbox-info">
            <input
              type="checkbox"
              id="cbFileUpload"
              className="custom-control-input"
              name="fileUpload"
              checked={adminAppContainer.state.fileUpload}
              onChange={(e) => {
                adminAppContainer.changeFileUpload(e.target.checked);
              }}
            />
            <label
              className="custom-control-label"
              htmlFor="cbFileUpload"
            >
              {t('admin:app_setting.enable_files_except_image')}
            </label>
          </div>

          <p className="form-text text-muted">
            {t('admin:app_setting.attach_enable')}
          </p>
        </div>
      </div>

      <AdminUpdateButtonRow onClick={submitHandler} disabled={adminAppContainer.state.retrieveError != null} />
    </React.Fragment>
  );

};


/**
 * Wrapper component for using unstated
 */
const AppSettingWrapper = withUnstatedContainers(AppSetting, [AdminAppContainer]);

AppSetting.propTypes = {
  adminAppContainer: PropTypes.instanceOf(AdminAppContainer).isRequired,
};


export default AppSettingWrapper;
