import React from 'react';

import { useTranslation } from 'next-i18next';
import PropTypes from 'prop-types';


import AdminGeneralSecurityContainer from '~/client/services/AdminGeneralSecurityContainer';
import AdminLocalSecurityContainer from '~/client/services/AdminLocalSecurityContainer';
import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { useIsMailerSetup } from '~/stores/context';

import { withUnstatedContainers } from '../../UnstatedUtils';

class LocalSecuritySettingContents extends React.Component {

  constructor(props) {
    super(props);

    this.onClickSubmit = this.onClickSubmit.bind(this);
  }

  async onClickSubmit() {
    const { t, adminGeneralSecurityContainer, adminLocalSecurityContainer } = this.props;
    try {
      await adminLocalSecurityContainer.updateLocalSecuritySetting();
      await adminGeneralSecurityContainer.retrieveSetupStratedies();
      toastSuccess(t('security_settings.updated_general_security_setting'));
    }
    catch (err) {
      toastError(err);
    }
  }

  render() {
    const {
      t,
      adminGeneralSecurityContainer,
      adminLocalSecurityContainer,
      isMailerSetup,
    } = this.props;
    const { registrationMode, isPasswordResetEnabled, isEmailAuthenticationEnabled } = adminLocalSecurityContainer.state;
    const { isLocalEnabled } = adminGeneralSecurityContainer.state;

    return (
      <>
        {adminLocalSecurityContainer.state.retrieveError != null && (
          <div className="alert alert-danger">
            <p>
              {t('Error occurred')} : {adminLocalSecurityContainer.state.retrieveError}
            </p>
          </div>
        )}
        <h2 className="alert-anchor border-bottom">{t('security_settings.Local.name')}</h2>

        {!isMailerSetup && (
          <div className="row">
            <div className="col-12">
              <div className="alert alert-danger">
                <span>{t('security_settings.Local.need_complete_mail_setting_warning')}</span>
                <a href="/admin/app#mail-settings"> <i className="fa fa-link"></i> {t('admin:app_setting.mail_settings')}</a>
              </div>
            </div>
          </div>
        )}

        {adminLocalSecurityContainer.state.useOnlyEnvVars && (
          <p
            className="alert alert-info"
            // eslint-disable-next-line max-len
            dangerouslySetInnerHTML={{
              __html: t('security_settings.Local.note for the only env option', { env: 'LOCAL_STRATEGY_USES_ONLY_ENV_VARS_FOR_SOME_OPTIONS' }),
            }}
          />
        )}

        <div className="row mb-5">
          <div className="col-6 offset-3">
            <div className="custom-control custom-switch custom-checkbox-success">
              <input
                type="checkbox"
                className="custom-control-input"
                id="isLocalEnabled"
                checked={isLocalEnabled}
                onChange={() => adminGeneralSecurityContainer.switchIsLocalEnabled()}
                disabled={adminLocalSecurityContainer.state.useOnlyEnvVars}
              />
              <label className="custom-control-label" htmlFor="isLocalEnabled">
                {t('security_settings.Local.enable_local')}
              </label>
            </div>
            {!adminGeneralSecurityContainer.state.setupStrategies.includes('local') && isLocalEnabled && (
              <div className="badge badge-warning">{t('security_settings.setup_is_not_yet_complete')}</div>
            )}
          </div>
        </div>

        {isLocalEnabled && (
          <>
            <h3 className="border-bottom">{t('security_settings.configuration')}</h3>

            <div className="row">
              <div className="col-12 col-md-3 text-left text-md-right py-2">
                <strong>{t('Register limitation')}</strong>
              </div>
              <div className="col-12 col-md-6">
                <div className="dropdown">
                  <button
                    className="btn btn-outline-secondary dropdown-toggle"
                    type="button"
                    id="dropdownMenuButton"
                    data-toggle="dropdown"
                    aria-haspopup="true"
                    aria-expanded="true"
                  >
                    {registrationMode === 'Open' && t('security_settings.registration_mode.open')}
                    {registrationMode === 'Restricted' && t('security_settings.registration_mode.restricted')}
                    {registrationMode === 'Closed' && t('security_settings.registration_mode.closed')}
                  </button>
                  <div className="dropdown-menu" aria-labelledby="dropdownMenuButton">
                    <button
                      className="dropdown-item"
                      type="button"
                      onClick={() => {
                        adminLocalSecurityContainer.changeRegistrationMode('Open');
                      }}
                    >
                      {t('security_settings.registration_mode.open')}
                    </button>
                    <button
                      className="dropdown-item"
                      type="button"
                      onClick={() => {
                        adminLocalSecurityContainer.changeRegistrationMode('Restricted');
                      }}
                    >
                      {t('security_settings.registration_mode.restricted')}
                    </button>
                    <button
                      className="dropdown-item"
                      type="button"
                      onClick={() => {
                        adminLocalSecurityContainer.changeRegistrationMode('Closed');
                      }}
                    >
                      {t('security_settings.registration_mode.closed')}
                    </button>
                  </div>
                </div>

                <p className="form-text text-muted small">{t('security_settings.Register limitation desc')}</p>
              </div>
            </div>
            <div className="row">
              <div className="col-12 col-md-3 text-left text-md-right">
                <strong dangerouslySetInnerHTML={{ __html: t('security_settings.The whitelist of registration permission E-mail address') }} />
              </div>
              <div className="col-12 col-md-6">
                <textarea
                  className="form-control"
                  type="textarea"
                  name="registrationWhiteList"
                  defaultValue={adminLocalSecurityContainer.state.registrationWhiteList.join('\n')}
                  onChange={e => adminLocalSecurityContainer.changeRegistrationWhiteList(e.target.value)}
                />
                <p className="form-text text-muted small">
                  {t('security_settings.restrict_emails')}
                  <br />
                  {t('security_settings.for_example')}
                  <code>@growi.org</code>
                  {t('security_settings.in_this_case')}
                  <br />
                  {t('security_settings.insert_single')}
                </p>
              </div>
            </div>

            <div className="row">
              <label className="col-12 col-md-3 text-left text-md-right  col-form-label">{t('security_settings.Local.password_reset_by_users')}</label>
              <div className="col-12 col-md-6">
                <div className="custom-control custom-switch custom-checkbox-success">
                  <input
                    type="checkbox"
                    className="custom-control-input"
                    id="isPasswordResetEnabled"
                    checked={isPasswordResetEnabled}
                    onChange={() => adminLocalSecurityContainer.switchIsPasswordResetEnabled()}
                  />
                  <label className="custom-control-label" htmlFor="isPasswordResetEnabled">
                    {t('security_settings.Local.enable_password_reset_by_users')}
                  </label>
                </div>
                <p className="form-text text-muted small">
                  {t('security_settings.Local.password_reset_desc')}
                </p>
              </div>
            </div>

            <div className="row">
              <label className="col-12 col-md-3 text-left text-md-right  col-form-label">{t('security_settings.Local.email_authentication')}</label>
              <div className="col-12 col-md-6">
                <div className="custom-control custom-switch custom-checkbox-success">
                  <input
                    type="checkbox"
                    className="custom-control-input"
                    id="isEmailAuthenticationEnabled"
                    checked={isEmailAuthenticationEnabled}
                    onChange={() => adminLocalSecurityContainer.switchIsEmailAuthenticationEnabled()}
                  />
                  <label className="custom-control-label" htmlFor="isEmailAuthenticationEnabled">
                    {t('security_settings.Local.enable_email_authentication')}
                  </label>
                </div>
                {!isMailerSetup && (
                  <div className="alert alert-warning p-1 my-1 small d-inline-block">
                    <span>{t('security_settings.Local.please_enable_mailer')}</span>
                    <a href="/admin/app#mail-settings"> <i className="fa fa-link"></i> {t('app_setting.mail_settings')}</a>
                  </div>
                )}
                <p className="form-text text-muted small">
                  {t('security_settings.Local.enable_email_authentication_desc')}
                </p>
              </div>
            </div>

            <div className="row my-3">
              <div className="offset-3 col-6">
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={adminLocalSecurityContainer.state.retrieveError != null}
                  onClick={this.onClickSubmit}
                >
                  {t('Update')}
                </button>
              </div>
            </div>
          </>
        )}
      </>
    );
  }

}

LocalSecuritySettingContents.propTypes = {
  t: PropTypes.func.isRequired, // i18next
  adminGeneralSecurityContainer: PropTypes.instanceOf(AdminGeneralSecurityContainer).isRequired,
  adminLocalSecurityContainer: PropTypes.instanceOf(AdminLocalSecurityContainer).isRequired,
};

const LocalSecuritySettingContentsWrapperFC = (props) => {
  const { t } = useTranslation('admin');
  const { data: isMailerSetup } = useIsMailerSetup();
  return <LocalSecuritySettingContents t={t} {...props} isMailerSetup={isMailerSetup ?? false} />;
};

const LocalSecuritySettingContentsWrapper = withUnstatedContainers(LocalSecuritySettingContentsWrapperFC, [
  AdminGeneralSecurityContainer,
  AdminLocalSecurityContainer,
]);

export default LocalSecuritySettingContentsWrapper;
