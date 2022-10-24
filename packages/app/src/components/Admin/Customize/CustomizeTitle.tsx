import React, { FC, useState } from 'react';

import { useTranslation } from 'next-i18next';
import { Card, CardBody } from 'reactstrap';

import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { apiv3Put } from '~/client/util/apiv3-client';
import { useCustomizeTitle } from '~/stores/context';

import AdminUpdateButtonRow from '../Common/AdminUpdateButtonRow';

export const CustomizeTitle: FC = () => {

  const { t } = useTranslation('admin');

  const { data: customizeTitle } = useCustomizeTitle();

  const [currentCustomizeTitle, setCrrentCustomizeTitle] = useState(customizeTitle);

  const onClickSubmit = async() => {
    try {
      await apiv3Put('/customize-setting/customize-title', {
        customizeTitle: currentCustomizeTitle,
      });
      toastSuccess(t('toaster.update_successed', { target: t('admin:customize_settings.custom_title'), ns: 'commons' }));
    }
    catch (err) {
      toastError(err);
    }
  };

  return (
    <React.Fragment>
      <div className="row">
        <div className="col-12">
          <h2 className="admin-setting-header">{t('admin:customize_settings.custom_title')}</h2>
        </div>

        <div className="col-12">
          <Card className="card well">
            <CardBody className="px-0 py-2">
              {/* eslint-disable react/no-danger */}
              <p dangerouslySetInnerHTML={{ __html: t('admin:customize_settings.custom_title_detail') }} />
              <ul>
                <li>
                  <span dangerouslySetInnerHTML={{ __html: t('admin:customize_settings.custom_title_detail_placeholder1') }} />
                </li>
                <li>
                  <span dangerouslySetInnerHTML={{ __html: t('admin:customize_settings.custom_title_detail_placeholder2') }} />
                </li>
                <li>
                  <span dangerouslySetInnerHTML={{ __html: t('admin:customize_settings.custom_title_detail_placeholder3') }} />
                </li>
              </ul>
              {/* eslint-enable react/no-danger */}
            </CardBody>
          </Card>
        </div>

        {/* TODO i18n */}
        <div className="form-text text-muted col-12">
            Default Value: <code>&#123;&#123;pagename&#125;&#125; - &#123;&#123;sitename&#125;&#125;</code>
          <br />
            Default Output Example: <code className="xml">&lt;title&gt;Page name - My GROWI&lt;&#047;title&gt;</code>
        </div>
        <div className="form-group col-12">
          <input
            className="form-control"
            defaultValue={currentCustomizeTitle}
            onChange={(e) => { setCrrentCustomizeTitle(e.target.value) }}
          />
        </div>
        <div className="col-12">
          <AdminUpdateButtonRow onClick={onClickSubmit} disabled={false} />
        </div>
      </div>
    </React.Fragment>
  );
};
