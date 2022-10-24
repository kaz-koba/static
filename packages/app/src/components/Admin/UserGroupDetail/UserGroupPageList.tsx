import React, { useEffect, useState, useCallback } from 'react';

import { useTranslation } from 'next-i18next';

import { toastError } from '~/client/util/apiNotification';
import { apiv3Get } from '~/client/util/apiv3-client';
import { IPageHasId } from '~/interfaces/page';

import { PageListItemS } from '../../PageList/PageListItemS';
import PaginationWrapper from '../../PaginationWrapper';

const pagingLimit = 10;

type Props = {
  userGroupId: string,
  relatedPages?: IPageHasId[],
}

const UserGroupPageList = (props: Props): JSX.Element => {
  const { t } = useTranslation('admin');
  const { userGroupId, relatedPages } = props;

  const [currentPages, setCurrentPages] = useState<IPageHasId[]>([]);
  const [activePage, setActivePage] = useState(1);
  const [total, setTotal] = useState(0);

  const handlePageChange = useCallback(async(pageNum) => {
    const offset = (pageNum - 1) * pagingLimit;

    try {
      const res = await apiv3Get(`/user-groups/${userGroupId}/pages`, {
        limit: pagingLimit,
        offset,
      });
      const { total, pages } = res.data;

      setTotal(total);
      setActivePage(pageNum);
      setCurrentPages(pages);
    }
    catch (err) {
      toastError(err);
    }
  }, [userGroupId]);

  useEffect(() => {
    handlePageChange(activePage);
  }, [activePage, handlePageChange]);

  return (
    <>
      <ul className="page-list-ul page-list-ul-flat mb-3">
        {currentPages.map(page => <li key={page._id}><PageListItemS page={page} /></li>)}
      </ul>
      {relatedPages != null && relatedPages.length === 0 ? <p>{t('user_group_management.no_pages')}</p> : (
        <PaginationWrapper
          activePage={activePage}
          changePage={handlePageChange}
          totalItemsCount={total}
          pagingLimit={pagingLimit}
          align="center"
          size="sm"
        />
      )}
    </>
  );
};

export default UserGroupPageList;
