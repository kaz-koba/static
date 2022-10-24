import React from 'react';

import { useTranslation } from 'next-i18next';

export const AdminNotFoundPage = (): JSX.Element => {
  const { t } = useTranslation();

  return (
    <h1 className="title">{t('not_found_page.page_not_exist')}</h1>
  );
};
