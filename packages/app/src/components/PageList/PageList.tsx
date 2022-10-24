import React from 'react';

import { useTranslation } from 'next-i18next';

import { IPageInfoForEntity, IPageWithMeta } from '~/interfaces/page';
import { OnDeletedFunction, OnPutBackedFunction } from '~/interfaces/ui';

import { ForceHideMenuItems } from '../Common/Dropdown/PageItemControl';

import { PageListItemL } from './PageListItemL';

import styles from './PageList.module.scss';

type Props<M extends IPageInfoForEntity> = {
  pages: IPageWithMeta<M>[],
  isEnableActions?: boolean,
  forceHideMenuItems?: ForceHideMenuItems,
  onPagesDeleted?: OnDeletedFunction,
  onPagePutBacked?: OnPutBackedFunction,
}

const PageList = (props: Props<IPageInfoForEntity>): JSX.Element => {
  const { t } = useTranslation();
  const {
    pages, isEnableActions, forceHideMenuItems, onPagesDeleted, onPagePutBacked,
  } = props;

  if (pages == null) {
    return (
      <div className="wiki">
        <div className="text-muted text-center">
          <i className="fa fa-2x fa-spinner fa-pulse mr-1"></i>
        </div>
      </div>
    );
  }

  const pageList = pages.map(page => (
    <PageListItemL
      key={page.data._id}
      page={page}
      isEnableActions={isEnableActions}
      forceHideMenuItems={forceHideMenuItems}
      onPageDeleted={onPagesDeleted}
      onPagePutBacked={onPagePutBacked}
    />
  ));

  if (pageList.length === 0) {
    return (
      <div className="mt-2">
        <p>{t('custom_navigation.no_page_list')}</p>
      </div>
    );
  }

  return (
    <div className={`page-list ${styles['page-list']}`}>
      <ul className="page-list-ul list-group list-group-flush">
        {pageList}
      </ul>
    </div>
  );
};

export default PageList;
