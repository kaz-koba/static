import React from 'react';

import { useTranslation } from 'react-i18next';

import { useSWRxCurrentPage } from '~/stores/page';


export const PageGrantAlert = (): JSX.Element => {
  const { t } = useTranslation();
  const { data: pageData } = useSWRxCurrentPage();

  if (pageData == null || pageData.grant == null || pageData.grant === 1) {
    return <></>;
  }

  const renderAlertContent = () => {
    const getGrantLabel = () => {
      if (pageData.grant === 2) {
        return (
          <>
            <i className="icon-fw icon-link"></i><strong>{t('Anyone with the link')} only</strong>
          </>
        );
      }
      if (pageData.grant === 4) {
        return (
          <>
            <i className="icon-fw icon-lock"></i><strong>{t('Only me')} only</strong>
          </>
        );
      }
      if (pageData.grant === 5) {
        return (
          <>
            <i className="icon-fw icon-organization"></i><strong>{pageData.grantedGroup.name} only</strong>
          </>
        );
      }
    };
    return (
      <>
        {getGrantLabel()} ({t('Browsing of this page is restricted')})
      </>
    );
  };


  return (
    <p className="alert alert-primary py-3 px-4">
      {renderAlertContent()}
    </p>
  );
};
