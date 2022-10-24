
import React from 'react';

import { useTranslation } from 'next-i18next';

import AdminAppContainer from '~/client/services/AdminAppContainer';

import { withUnstatedContainers } from '../../UnstatedUtils';


type Props = {
  adminAppContainer: AdminAppContainer,
}

const SmtpSetting = (props: Props) => {
  const { t } = useTranslation();
  const { adminAppContainer } = props;

  return (
    <React.Fragment>
      <div id="mail-smtp" className="tab-pane active mt-5">
        <div className="row form-group">
          <label className="text-left text-md-right col-md-3 col-form-label">
            {t('admin:app_setting.host')}
          </label>
          <div className="col-md-6">
            <input
              className="form-control"
              type="text"
              defaultValue={adminAppContainer.state.smtpHost || ''}
              onChange={(e) => { adminAppContainer.changeSmtpHost(e.target.value) }}
            />
          </div>
        </div>

        <div className="row form-group">
          <label className="text-left text-md-right col-md-3 col-form-label">
            {t('admin:app_setting.port')}
          </label>
          <div className="col-md-6">
            <input
              className="form-control"
              defaultValue={adminAppContainer.state.smtpPort || ''}
              onChange={(e) => { adminAppContainer.changeSmtpPort(e.target.value) }}
            />
          </div>
        </div>

        <div className="row form-group">
          <label className="text-left text-md-right col-md-3 col-form-label">
            {t('admin:app_setting.user')}
          </label>
          <div className="col-md-6">
            <input
              className="form-control"
              type="text"
              defaultValue={adminAppContainer.state.smtpUser || ''}
              onChange={(e) => { adminAppContainer.changeSmtpUser(e.target.value) }}
            />
          </div>
        </div>

        <div className="row form-group">
          <label className="text-left text-md-right col-md-3 col-form-label">
            {t('Password')}
          </label>
          <div className="col-md-6">
            <input
              className="form-control"
              type="password"
              defaultValue={adminAppContainer.state.smtpPassword || ''}
              onChange={(e) => { adminAppContainer.changeSmtpPassword(e.target.value) }}
            />
          </div>
        </div>
      </div>
    </React.Fragment>
  );
};

/**
 * Wrapper component for using unstated
 */
const SmtpSettingWrapper = withUnstatedContainers(SmtpSetting, [AdminAppContainer]);
export default SmtpSettingWrapper;
