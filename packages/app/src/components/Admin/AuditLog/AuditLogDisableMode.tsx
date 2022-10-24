import React, { FC } from 'react';

import { useTranslation } from 'react-i18next';

export const AuditLogDisableMode: FC = () => {
  const { t } = useTranslation('admin');

  return (
    <div id="content-main" className="content-main container-lg">
      <div className="container">
        <div className="row justify-content-md-center">
          <div className="col-md-6 mt-5">
            <div className="text-center">
              <h1><i className="icon-exclamation large"></i></h1>
              <h1 className="text-center">{t('audit_log_management.audit_log')}</h1>
              <h3
                // eslint-disable-next-line react/no-danger
                dangerouslySetInnerHTML={{ __html: t('audit_log_management.disable_mode_explain') }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
