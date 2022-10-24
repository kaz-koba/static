import React, {
  forwardRef, useState,
  ForwardRefRenderFunction, memo, useCallback, useImperativeHandle, useRef, useEffect,
} from 'react';


import { DevidedPagePath } from '@growi/core';
import { UserPicture, PageListMeta } from '@growi/ui';
import { format } from 'date-fns';
import { useTranslation } from 'next-i18next';
import Link from 'next/link';
import Clamp from 'react-multiline-clamp';
import { CustomInput } from 'reactstrap';
import urljoin from 'url-join';


import { ISelectable } from '~/client/interfaces/selectable-all';
import { bookmark, unbookmark } from '~/client/services/page-operation';
import {
  IPageInfoAll, isIPageInfoForListing, isIPageInfoForEntity, IPageWithMeta, IPageInfoForListing,
} from '~/interfaces/page';
import { IPageSearchMeta, IPageWithSearchMeta, isIPageSearchMeta } from '~/interfaces/search';
import {
  OnDuplicatedFunction, OnRenamedFunction, OnDeletedFunction, OnPutBackedFunction,
} from '~/interfaces/ui';
import LinkedPagePath from '~/models/linked-page-path';
import {
  usePageRenameModal, usePageDuplicateModal, usePageDeleteModal, usePutBackPageModal,
} from '~/stores/modal';
import { useIsDeviceSmallerThanLg } from '~/stores/ui';

import { useSWRxPageInfo } from '../../stores/page';
import { ForceHideMenuItems, PageItemControl } from '../Common/Dropdown/PageItemControl';
import PagePathHierarchicalLink from '../PagePathHierarchicalLink';

type Props = {
  page: IPageWithSearchMeta | IPageWithMeta<IPageInfoForListing & IPageSearchMeta>,
  isSelected?: boolean, // is item selected(focused)
  isEnableActions?: boolean,
  forceHideMenuItems?: ForceHideMenuItems,
  showPageUpdatedTime?: boolean, // whether to show page's updated time at the top-right corner of item
  onCheckboxChanged?: (isChecked: boolean, pageId: string) => void,
  onClickItem?: (pageId: string) => void,
  onPageDuplicated?: OnDuplicatedFunction,
  onPageRenamed?: OnRenamedFunction,
  onPageDeleted?: OnDeletedFunction,
  onPagePutBacked?: OnPutBackedFunction,
}

const PageListItemLSubstance: ForwardRefRenderFunction<ISelectable, Props> = (props: Props, ref): JSX.Element => {
  const {
    page: { data: pageData, meta: pageMeta }, isSelected, isEnableActions,
    forceHideMenuItems,
    showPageUpdatedTime,
    onClickItem, onCheckboxChanged, onPageDuplicated, onPageRenamed, onPageDeleted, onPagePutBacked,
  } = props;

  const [likerCount, setLikerCount] = useState(pageData.liker.length);
  const [bookmarkCount, setBookmarkCount] = useState(pageMeta && pageMeta.bookmarkCount ? pageMeta.bookmarkCount : 0);

  const { t } = useTranslation();

  const inputRef = useRef<HTMLInputElement>(null);

  // publish ISelectable methods
  useImperativeHandle(ref, () => ({
    select: () => {
      const input = inputRef.current;
      if (input != null) {
        input.checked = true;
      }
    },
    deselect: () => {
      const input = inputRef.current;
      if (input != null) {
        input.checked = false;
      }
    },
  }));

  const { data: isDeviceSmallerThanLg } = useIsDeviceSmallerThanLg();
  const { open: openDuplicateModal } = usePageDuplicateModal();
  const { open: openRenameModal } = usePageRenameModal();
  const { open: openDeleteModal } = usePageDeleteModal();
  const { open: openPutBackPageModal } = usePutBackPageModal();

  const shouldFetch = isSelected && (pageData != null || pageMeta != null);
  const { data: pageInfo } = useSWRxPageInfo(shouldFetch ? pageData?._id : null);

  const elasticSearchResult = isIPageSearchMeta(pageMeta) ? pageMeta.elasticSearchResult : null;
  const revisionShortBody = isIPageInfoForListing(pageMeta) ? pageMeta.revisionShortBody : null;

  const dPagePath: DevidedPagePath = new DevidedPagePath(pageData.path, false);
  const linkedPagePathFormer = new LinkedPagePath(dPagePath.former);

  const dPagePathHighlighted: DevidedPagePath = new DevidedPagePath(elasticSearchResult?.highlightedPath || pageData.path, true);
  const linkedPagePathHighlightedFormer = new LinkedPagePath(dPagePathHighlighted.former);
  const linkedPagePathHighlightedLatter = new LinkedPagePath(dPagePathHighlighted.latter);

  const lastUpdateDate = format(new Date(pageData.updatedAt), 'yyyy/MM/dd HH:mm:ss');

  useEffect(() => {
    if (isIPageInfoForEntity(pageInfo)) {
      // likerCount
      setLikerCount(pageInfo.likerIds?.length ?? 0);
      // bookmarkCount
      setBookmarkCount(pageInfo.bookmarkCount ?? 0);
    }
  }, [pageInfo]);

  // click event handler
  const clickHandler = useCallback(() => {
    // do nothing if mobile
    if (isDeviceSmallerThanLg) {
      return;
    }

    if (onClickItem != null) {
      onClickItem(pageData._id);
    }
  }, [isDeviceSmallerThanLg, onClickItem, pageData._id]);

  const bookmarkMenuItemClickHandler = async(_pageId: string, _newValue: boolean): Promise<void> => {
    const bookmarkOperation = _newValue ? bookmark : unbookmark;
    await bookmarkOperation(_pageId);
  };

  const duplicateMenuItemClickHandler = useCallback(() => {
    const page = {
      pageId: pageData._id,
      path: pageData.path,
    };
    openDuplicateModal(page, { onDuplicated: onPageDuplicated });
  }, [onPageDuplicated, openDuplicateModal, pageData._id, pageData.path]);

  const renameMenuItemClickHandler = useCallback((_id: string, pageInfo: IPageInfoAll | undefined) => {
    const page = { data: pageData, meta: pageInfo };
    openRenameModal(page, { onRenamed: onPageRenamed });
  }, [pageData, onPageRenamed, openRenameModal]);


  const deleteMenuItemClickHandler = useCallback((_id: string, pageInfo: IPageInfoAll | undefined) => {
    const pageToDelete = { data: pageData, meta: pageInfo };

    // open modal
    openDeleteModal([pageToDelete], { onDeleted: onPageDeleted });
  }, [pageData, openDeleteModal, onPageDeleted]);

  const revertMenuItemClickHandler = useCallback(() => {
    const { _id: pageId, path } = pageData;
    openPutBackPageModal({ pageId, path }, { onPutBacked: onPagePutBacked });
  }, [onPagePutBacked, openPutBackPageModal, pageData]);

  const styleListGroupItem = (!isDeviceSmallerThanLg && onClickItem != null) ? 'list-group-item-action' : '';
  // background color of list item changes when class "active" exists under 'list-group-item'
  const styleActive = !isDeviceSmallerThanLg && isSelected ? 'active' : '';

  const shouldDangerouslySetInnerHTMLForPaths = elasticSearchResult != null && elasticSearchResult.highlightedPath != null;

  const canRenderESSnippet = elasticSearchResult != null && elasticSearchResult.snippet != null;
  const canRenderRevisionSnippet = revisionShortBody != null;

  return (
    <li
      key={pageData._id}
      className={`list-group-item d-flex align-items-center px-3 px-md-1 ${styleListGroupItem} ${styleActive}`}
      data-testid="page-list-item-L"
      onClick={clickHandler}
    >
      <div className="text-break w-100">
        <div className="d-flex">
          {/* checkbox */}
          {onCheckboxChanged != null && (
            <div className="d-flex align-items-center justify-content-center">
              <CustomInput
                type="checkbox"
                id={`cbSelect-${pageData._id}`}
                data-testid="cb-select"
                innerRef={inputRef}
                onChange={(e) => { onCheckboxChanged(e.target.checked, pageData._id) }}
              />
            </div>
          )}

          <div className="flex-grow-1 px-2 px-md-4">
            <div className="d-flex justify-content-between">
              {/* page path */}
              <PagePathHierarchicalLink
                linkedPagePath={linkedPagePathFormer}
                linkedPagePathByHtml={linkedPagePathHighlightedFormer}
              />
              { showPageUpdatedTime && (
                <span className="page-list-updated-at text-muted">Last update: {lastUpdateDate}</span>
              ) }
            </div>
            <div className="d-flex align-items-center mb-1">
              {/* Picture */}
              <span className="mr-2 d-none d-md-block">
                <UserPicture user={pageData.lastUpdateUser} size="md" />
              </span>
              {/* page title */}
              <Clamp lines={1}>
                <span className="h5 mb-0">
                  {/* Use permanent links to care for pages with the same name (Cannot use page path url) */}
                  <span className="grw-page-path-hierarchical-link text-break">
                    <Link href={encodeURI(urljoin('/', pageData._id))} prefetch={false}>
                      {shouldDangerouslySetInnerHTMLForPaths
                        ? (
                          <a
                            className="page-segment"
                            // eslint-disable-next-line react/no-danger
                            dangerouslySetInnerHTML={{ __html: linkedPagePathHighlightedLatter.pathName }}
                          >
                          </a>
                        )
                        : <a className="page-segment">{linkedPagePathHighlightedLatter.pathName}</a>
                      }
                    </Link>
                  </span>
                </span>
              </Clamp>

              {/* page meta */}
              <div className="d-none d-md-flex py-0 px-1 ml-2 text-nowrap">
                <PageListMeta page={pageData} likerCount={likerCount} bookmarkCount={bookmarkCount} shouldSpaceOutIcon />
              </div>

              {/* doropdown icon includes page control buttons */}
              <div className="ml-auto">
                <PageItemControl
                  alignRight
                  pageId={pageData._id}
                  pageInfo={isIPageInfoForListing(pageMeta) ? pageMeta : undefined}
                  isEnableActions={isEnableActions}
                  forceHideMenuItems={forceHideMenuItems}
                  onClickBookmarkMenuItem={bookmarkMenuItemClickHandler}
                  onClickRenameMenuItem={renameMenuItemClickHandler}
                  onClickDuplicateMenuItem={duplicateMenuItemClickHandler}
                  onClickDeleteMenuItem={deleteMenuItemClickHandler}
                  onClickRevertMenuItem={revertMenuItemClickHandler}
                />
              </div>
            </div>
            <div className="page-list-snippet py-1">
              <Clamp lines={2}>
                { elasticSearchResult != null && elasticSearchResult.snippet != null && (
                  // eslint-disable-next-line react/no-danger
                  <div dangerouslySetInnerHTML={{ __html: elasticSearchResult.snippet }}></div>
                ) }
                { revisionShortBody != null && (
                  <div>{revisionShortBody}</div>
                ) }
                {
                  !canRenderESSnippet && !canRenderRevisionSnippet && (
                    <>
                      <i className="icon-exclamation p-1"></i>
                      {t('not_allowed_to_see_this_page')}
                    </>
                  )
                }
              </Clamp>
            </div>
          </div>
        </div>
        {/* TODO: adjust snippet position */}
      </div>
    </li>
  );
};

export const PageListItemL = memo(forwardRef(PageListItemLSubstance));
