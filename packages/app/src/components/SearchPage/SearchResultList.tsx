import React, {
  forwardRef,
  ForwardRefRenderFunction, useCallback, useImperativeHandle, useRef,
} from 'react';

import { useTranslation } from 'next-i18next';

import { ISelectable, ISelectableAll } from '~/client/interfaces/selectable-all';
import { toastSuccess } from '~/client/util/apiNotification';
import {
  IPageInfoForListing, IPageWithMeta, isIPageInfoForListing,
} from '~/interfaces/page';
import { IPageSearchMeta, IPageWithSearchMeta } from '~/interfaces/search';
import { OnDuplicatedFunction, OnRenamedFunction, OnDeletedFunction } from '~/interfaces/ui';
import { useIsGuestUser } from '~/stores/context';
import { useSWRxPageInfoForList, usePageTreeTermManager } from '~/stores/page-listing';
import { useFullTextSearchTermManager } from '~/stores/search';

import { ForceHideMenuItems } from '../Common/Dropdown/PageItemControl';
import { PageListItemL } from '../PageList/PageListItemL';


type Props = {
  pages: IPageWithSearchMeta[],
  selectedPageId?: string,
  forceHideMenuItems?: ForceHideMenuItems,
  onPageSelected?: (page?: IPageWithSearchMeta) => void,
  onCheckboxChanged?: (isChecked: boolean, pageId: string) => void,
}

const SearchResultListSubstance: ForwardRefRenderFunction<ISelectableAll, Props> = (props:Props, ref) => {
  const {
    pages, selectedPageId,
    forceHideMenuItems,
    onPageSelected,
  } = props;

  const { t } = useTranslation();

  const pageIdsWithNoSnippet = pages
    .filter(page => (page.meta?.elasticSearchResult?.snippet?.length ?? 0) === 0)
    .map(page => page.data._id);

  const { data: isGuestUser } = useIsGuestUser();
  const { data: idToPageInfo } = useSWRxPageInfoForList(pageIdsWithNoSnippet, null, true, true);

  // for mutation
  const { advance: advancePt } = usePageTreeTermManager();
  const { advance: advanceFts } = useFullTextSearchTermManager();

  const itemsRef = useRef<(ISelectable|null)[]>([]);

  // publish selectAll()
  useImperativeHandle(ref, () => ({
    selectAll: () => {
      const items = itemsRef.current;
      if (items != null) {
        items.forEach(item => item != null && item.select());
      }
    },
    deselectAll: () => {
      const items = itemsRef.current;
      if (items != null) {
        items.forEach(item => item != null && item.deselect());
      }
    },
  }));

  const clickItemHandler = useCallback((pageId: string) => {
    if (onPageSelected != null) {
      const selectedPage = pages.find(page => page.data._id === pageId);
      onPageSelected(selectedPage);
    }
  }, [onPageSelected, pages]);

  let injectedPages: (IPageWithSearchMeta | IPageWithMeta<IPageInfoForListing & IPageSearchMeta>)[] | undefined;
  // inject data to list
  if (idToPageInfo != null) {
    injectedPages = pages.map((page) => {
      const pageInfo = idToPageInfo[page.data._id];

      if (!isIPageInfoForListing(pageInfo)) {
        // return as is
        return page;
      }

      return {
        data: page.data,
        meta: {
          ...page.meta,
          ...pageInfo,
        },
      } as IPageWithMeta<IPageInfoForListing & IPageSearchMeta>;
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const duplicatedHandler : OnDuplicatedFunction = (fromPath, toPath) => {
    toastSuccess(t('duplicated_pages', { fromPath }));

    advancePt();
    advanceFts();
  };

  const renamedHandler: OnRenamedFunction = (path) => {
    toastSuccess(t('renamed_pages', { path }));

    advancePt();
    advanceFts();
  };
  const deletedHandler: OnDeletedFunction = (pathOrPathsToDelete, isRecursively, isCompletely) => {
    if (typeof pathOrPathsToDelete !== 'string') {
      return;
    }

    const path = pathOrPathsToDelete;

    if (isCompletely) {
      toastSuccess(t('deleted_pages_completely', { path }));
    }
    else {
      toastSuccess(t('deleted_pages', { path }));
    }
    advancePt();
    advanceFts();
  };

  return (
    <ul data-testid="search-result-list" className="page-list-ul list-group list-group-flush">
      { (injectedPages ?? pages).map((page, i) => {
        return (
          <PageListItemL
            key={page.data._id}
            // eslint-disable-next-line no-return-assign
            ref={c => itemsRef.current[i] = c}
            page={page}
            isEnableActions={!isGuestUser}
            isSelected={page.data._id === selectedPageId}
            forceHideMenuItems={forceHideMenuItems}
            onClickItem={clickItemHandler}
            onCheckboxChanged={props.onCheckboxChanged}
            onPageDuplicated={duplicatedHandler}
            onPageRenamed={renamedHandler}
            onPageDeleted={deletedHandler}
          />
        );
      })}
    </ul>
  );

};

export const SearchResultList = forwardRef(SearchResultListSubstance);
