import React from 'react';

import { pathUtils } from '@growi/core';
import { useTranslation } from 'next-i18next';
import PropTypes from 'prop-types';
import urljoin from 'url-join';

import AdminGeneralSecurityContainer from '~/client/services/AdminGeneralSecurityContainer';
import AdminGoogleSecurityContainer from '~/client/services/AdminGoogleSecurityContainer';
import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { useSiteUrl } from '~/stores/context';

import { withUnstatedContainers } from '../../UnstatedUtils';

class GoogleSecurityManagementContents extends React.Component {

  constructor(props) {
    super(props);

    this.onClickSubmit = this.onClickSubmit.bind(this);
  }

  async onClickSubmit() {
    const { t, adminGoogleSecurityContainer, adminGeneralSecurityContainer } = this.props;

    try {
      await adminGoogleSecurityContainer.updateGoogleSetting();
      await adminGeneralSecurityContainer.retrieveSetupStratedies();
      toastSuccess(t('security_settings.OAuth.Google.updated_google'));
    }
    catch (err) {
      toastError(err);
    }
  }

  render() {
    const {
      t, adminGeneralSecurityContainer, adminGoogleSecurityContainer, siteUrl,
    } = this.props;
    const { isGoogleEnabled } = adminGeneralSecurityContainer.state;
    const googleCallbackUrl = urljoin(pathUtils.removeTrailingSlash(siteUrl), '/passport/google/callback');

    return (

      <React.Fragment>

        <h2 className="alert-anchor border-bottom">
          {t('security_settings.OAuth.Google.name')}
        </h2>

        {adminGoogleSecurityContainer.state.retrieveError != null && (
          <div className="alert alert-danger">
            <p>{t('Error occurred')} : {adminGoogleSecurityContainer.state.retrieveError}</p>
          </div>
        )}

        <div className="form-group row">
          <div className="col-6 offset-3">
            <div className="custom-control custom-switch custom-checkbox-success">
              <input
                id="isGoogleEnabled"
                className="custom-control-input"
                type="checkbox"
                checked={adminGeneralSecurityContainer.state.isGoogleEnabled || false}
                onChange={() => { adminGeneralSecurityContainer.switchIsGoogleOAuthEnabled() }}
              />
              <label className="custom-control-label" htmlFor="isGoogleEnabled">
                {t('security_settings.OAuth.Google.enable_google')}
              </label>
            </div>
            {(!adminGeneralSecurityContainer.state.setupStrategies.includes('google') && isGoogleEnabled)
              && <div className="badge badge-warning">{t('security_settings.setup_is_not_yet_complete')}</div>}
          </div>
        </div>

        <div className="row mb-5">
          <label className="col-12 col-md-3 text-left text-md-right py-2">{t('security_settings.callback_URL')}</label>
          <div className="col-12 col-md-6">
            <input
              className="form-control"
              type="text"
              value={googleCallbackUrl}
              readOnly
            />
            <p className="form-text text-muted small">{t('security_settings.desc_of_callback_URL', { AuthName: 'OAuth' })}</p>
            {(siteUrl == null || siteUrl === '') && (
              <div className="alert alert-danger">
                <i
                  className="icon-exclamation"
                  // eslint-disable-next-line max-len
                  dangerouslySetInnerHTML={{ __html: t('commons:alert.siteUrl_is_not_set', { link: `<a href="/admin/app">${t('commons:headers.app_settings')}<i class="icon-login"></i></a>` }) }}
                />
              </div>
            )}
          </div>
        </div>


        {isGoogleEnabled && (
          <React.Fragment>

            <h3 className="border-bottom">{t('security_settings.configuration')}</h3>

            <div className="row mb-5">
              <label htmlFor="googleClientId" className="col-3 text-right py-2">{t('security_settings.clientID')}</label>
              <div className="col-6">
                <input
                  className="form-control"
                  type="text"
                  name="googleClientId"
                  defaultValue={adminGoogleSecurityContainer.state.googleClientId || ''}
                  onChange={e => adminGoogleSecurityContainer.changeGoogleClientId(e.target.value)}
                />
                <p className="form-text text-muted">
                  <small dangerouslySetInnerHTML={{ __html: t('security_settings.Use env var if empty', { env: 'OAUTH_GOOGLE_CLIENT_ID' }) }} />
                </p>
              </div>
            </div>

            <div className="row mb-5">
              <label htmlFor="googleClientSecret" className="col-3 text-right py-2">{t('security_settings.client_secret')}</label>
              <div className="col-6">
                <input
                  className="form-control"
                  type="text"
                  name="googleClientSecret"
                  defaultValue={adminGoogleSecurityContainer.state.googleClientSecret || ''}
                  onChange={e => adminGoogleSecurityContainer.changeGoogleClientSecret(e.target.value)}
                />
                <p className="form-text text-muted">
                  <small dangerouslySetInnerHTML={{ __html: t('security_settings.Use env var if empty', { env: 'OAUTH_GOOGLE_CLIENT_SECRET' }) }} />
                </p>
              </div>
            </div>

            <div className="row mb-5">
              <div className="offset-3 col-6">
                <div className="custom-control custom-checkbox custom-checkbox-success">
                  <input
                    id="bindByUserNameGoogle"
                    className="custom-control-input"
                    type="checkbox"
                    checked={adminGoogleSecurityContainer.state.isSameEmailTreatedAsIdenticalUser || false}
                    onChange={() => { adminGoogleSecurityContainer.switchIsSameEmailTreatedAsIdenticalUser() }}
                  />
                  <label
                    className="custom-control-label"
                    htmlFor="bindByUserNameGoogle"
                    dangerouslySetInnerHTML={{ __html: t('security_settings.Treat email matching as identical') }}
                  />
                </div>
                <p className="form-text text-muted">
                  <small dangerouslySetInnerHTML={{ __html: t('security_settings.Treat email matching as identical_warn') }} />
                </p>
              </div>
            </div>

            <div className="row my-3">
              <div className="offset-3 col-5">
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={adminGoogleSecurityContainer.state.retrieveError != null}
                  onClick={this.onClickSubmit}
                >
                  {t('Update')}
                </button>
              </div>
            </div>

          </React.Fragment>
        )}

        <hr />

        <div style={{ minHeight: '300px' }}>
          <h4>
            <i className="icon-question" aria-hidden="true"></i>
            <a href="#collapseHelpForGoogleOauth" data-toggle="collapse"> {t('security_settings.OAuth.how_to.google')}</a>
          </h4>
          <ol id="collapseHelpForGoogleOauth" className="collapse">
            {/* eslint-disable-next-line max-len */}
            <li dangerouslySetInnerHTML={{ __html: t('security_settings.OAuth.Google.register_1', { link: '<a href="https://console.cloud.google.com/apis/credentials" target=_blank>Google Cloud Platform API Manager</a>' }) }} />
            <li dangerouslySetInnerHTML={{ __html: t('security_settings.OAuth.Google.register_2') }} />
            <li dangerouslySetInnerHTML={{ __html: t('security_settings.OAuth.Google.register_3') }} />
            <li dangerouslySetInnerHTML={{ __html: t('security_settings.OAuth.Google.register_4', { url: googleCallbackUrl }) }} />
            <li dangerouslySetInnerHTML={{ __html: t('security_settings.OAuth.Google.register_5') }} />
          </ol>
        </div>

      </React.Fragment>


    );
  }

}

const GoogleSecurityManagementContentsFc = (props) => {
  const { t } = useTranslation('admin');
  const { data: siteUrl } = useSiteUrl();
  return <GoogleSecurityManagementContents t={t} siteUrl={siteUrl} {...props} />;
};


GoogleSecurityManagementContents.propTypes = {
  t: PropTypes.func.isRequired, // i18next
  adminGeneralSecurityContainer: PropTypes.instanceOf(AdminGeneralSecurityContainer).isRequired,
  adminGoogleSecurityContainer: PropTypes.instanceOf(AdminGoogleSecurityContainer).isRequired,
  siteUrl: PropTypes.string,
};

const GoogleSecurityManagementContentsWrapper = withUnstatedContainers(GoogleSecurityManagementContentsFc, [
  AdminGeneralSecurityContainer,
  AdminGoogleSecurityContainer,
]);

export default GoogleSecurityManagementContentsWrapper;
