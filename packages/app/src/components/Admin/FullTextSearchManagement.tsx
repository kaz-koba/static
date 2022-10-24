import React from 'react';

import { useTranslation } from 'next-i18next';

import ElasticsearchManagement from './ElasticsearchManagement/ElasticsearchManagement';

const FullTextSearchManagement = (): JSX.Element => {
  const { t } = useTranslation();

  return (
    <div data-testid="admin-full-text-search">
      <h2> { t('full_text_search_management.elasticsearch_management') } </h2>
      <ElasticsearchManagement />
    </div>
  );
};

export default FullTextSearchManagement;
