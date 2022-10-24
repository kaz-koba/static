import React, { useEffect } from 'react';

import { useTranslation } from 'next-i18next';

import AdminAppContainer from '~/client/services/AdminAppContainer';
import { toastError } from '~/client/util/apiNotification';
import { useIsMaintenanceMode } from '~/stores/maintenanceMode';
import { toArrayIfNot } from '~/utils/array-utils';
import loggerFactory from '~/utils/logger';

import { withUnstatedContainers } from '../../UnstatedUtils';

import AppSetting from './AppSetting';
import FileUploadSetting from './FileUploadSetting';
import MailSetting from './MailSetting';
import { MaintenanceMode } from './MaintenanceMode';
import PluginSetting from './PluginSetting';
import SiteUrlSetting from './SiteUrlSetting';
import V5PageMigration from './V5PageMigration';


const logger = loggerFactory('growi:appSettings');

type Props = {
  adminAppContainer: AdminAppContainer,
}

const AppSettingsPageContents = (props: Props) => {
  const { t } = useTranslation('admin');
  const { adminAppContainer } = props;

  const { data: isMaintenanceMode } = useIsMaintenanceMode();

  const { isV5Compatible } = adminAppContainer.state;

  useEffect(() => {
    const fetchAppSettingsData = async() => {
      await adminAppContainer.retrieveAppSettingsData();
    };

    try {
      fetchAppSettingsData();
    }
    catch (err) {
      const errs = toArrayIfNot(err);
      toastError(errs);
      logger.error(errs);
    }
  }, [adminAppContainer]);

  return (
    <div data-testid="admin-app-settings">
      {
        // Alert message will be displayed in case that the GROWI is under maintenance
        isMaintenanceMode && (
          <div className="alert alert-danger alert-link" role="alert">
            <h3 className="alert-heading">
              {t('admin:maintenance_mode.maintenance_mode')}
            </h3>
            <p>
              {t('admin:maintenance_mode.description')}
            </p>
            <hr />
            <a className="btn-link" href="#maintenance-mode" rel="noopener noreferrer">
              <i className="fa fa-fw fa-arrow-down ml-1" aria-hidden="true"></i>
              <strong>{t('admin:maintenance_mode.end_maintenance_mode')}</strong>
            </a>
          </div>
        )
      }
      {
        !isV5Compatible
          && (
            <div className="row">
              <div className="col-lg-12">
                <h2 className="admin-setting-header">{t('V5 Page Migration')}</h2>
                <V5PageMigration />
              </div>
            </div>
          )
      }

      <div className="row">
        <div className="col-lg-12">
          <h2 className="admin-setting-header">{t('commons:headers.app_settings')}</h2>
          <AppSetting />
        </div>
      </div>

      <div className="row mt-5">
        <div className="col-lg-12">
          <h2 className="admin-setting-header">{t('Site URL settings')}</h2>
          <SiteUrlSetting />
        </div>
      </div>

      <div className="row mt-5">
        <div className="col-lg-12">
          <h2 className="admin-setting-header" id="mail-settings">{t('app_setting.mail_settings')}</h2>
          <MailSetting />
        </div>
      </div>

      <div className="row mt-5">
        <div className="col-lg-12">
          <h2 className="admin-setting-header">{t('admin:app_setting.file_upload_settings')}</h2>
          <FileUploadSetting />
        </div>
      </div>

      <div className="row mt-5">
        <div className="col-lg-12">
          <h2 className="admin-setting-header">{t('admin:app_setting.plugin_settings')}</h2>
          <PluginSetting />
        </div>
      </div>

      <div className="row">
        <div className="col-lg-12">
          <h2 className="admin-setting-header" id="maintenance-mode">{t('admin:maintenance_mode.maintenance_mode')}</h2>
          <MaintenanceMode />
        </div>
      </div>
    </div>
  );
};

/**
 * Wrapper component for using unstated
 */
const AppSettingsPageContentsWrapper = withUnstatedContainers(AppSettingsPageContents, [AdminAppContainer]);

export default AppSettingsPageContentsWrapper;
