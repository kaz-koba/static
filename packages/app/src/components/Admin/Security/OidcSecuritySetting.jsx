import React, { useEffect, useCallback } from 'react';

import PropTypes from 'prop-types';

import AdminOidcSecurityContainer from '~/client/services/AdminOidcSecurityContainer';
import { toastError } from '~/client/util/apiNotification';
import { toArrayIfNot } from '~/utils/array-utils';

import { withUnstatedContainers } from '../../UnstatedUtils';

import OidcSecurityManagementContents from './OidcSecuritySettingContents';

const OidcSecurityManagement = (props) => {
  const { adminOidcSecurityContainer } = props;

  const fetchOidcSecuritySettingsData = useCallback(async() => {
    try {
      await adminOidcSecurityContainer.retrieveSecurityData();
    }
    catch (err) {
      const errs = toArrayIfNot(err);
      toastError(errs);
    }
  }, [adminOidcSecurityContainer]);

  useEffect(() => {
    fetchOidcSecuritySettingsData();
  }, [adminOidcSecurityContainer, fetchOidcSecuritySettingsData]);

  return <OidcSecurityManagementContents />;
};

OidcSecurityManagement.propTypes = {
  adminOidcSecurityContainer: PropTypes.instanceOf(AdminOidcSecurityContainer).isRequired,
};

const OidcSecurityManagementWithUnstatedContainer = withUnstatedContainers(OidcSecurityManagement, [
  AdminOidcSecurityContainer,
]);

export default OidcSecurityManagementWithUnstatedContainer;
