
import React from 'react';

import AdminAppContainer from '~/client/services/AdminAppContainer';

import { withUnstatedContainers } from '../../UnstatedUtils';

type Props = {
  adminAppContainer: AdminAppContainer,
}

const SmtpSetting = (props: Props) => {
  const { adminAppContainer } = props;

  return (
    <React.Fragment>
      <div id="mail-smtp" className="tab-pane active mt-5">

        <div className="row form-group">
          <label className="text-left text-md-right col-md-3 col-form-label">
            Access key ID
          </label>
          <div className="col-md-6">
            <input
              className="form-control"
              type="text"
              defaultValue={adminAppContainer.state.sesAccessKeyId || ''}
              onChange={(e) => {
                adminAppContainer.changeSesAccessKeyId(e.target.value);
              }}
            />
          </div>
        </div>

        <div className="row form-group">
          <label className="text-left text-md-right col-md-3 col-form-label">
            Secret access key
          </label>
          <div className="col-md-6">
            <input
              className="form-control"
              type="text"
              defaultValue={adminAppContainer.state.sesSecretAccessKey || ''}
              onChange={(e) => {
                adminAppContainer.changeSesSecretAccessKey(e.target.value);
              }}
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
