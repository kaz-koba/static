import React, {
  useCallback, useMemo, useRef, useState, useEffect,
} from 'react';

import { useTranslation } from 'next-i18next';
import {
  UncontrolledButtonDropdown, DropdownToggle, DropdownMenu, DropdownItem, Modal, ModalHeader, ModalBody, ModalFooter,
} from 'reactstrap';

import { ISelectableAll, ISelectableAndIndeterminatable } from '~/client/interfaces/selectable-all';
import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { apiv3Post } from '~/client/util/apiv3-client';
import { V5ConversionErrCode } from '~/interfaces/errors/v5-conversion-error';
import { V5MigrationStatus } from '~/interfaces/page-listing-results';
import { IFormattedSearchResult } from '~/interfaces/search';
import { PageMigrationErrorData, SocketEventName } from '~/interfaces/websocket';
import { useCurrentUser } from '~/stores/context';
import {
  ILegacyPrivatePage, usePrivateLegacyPagesMigrationModal,
} from '~/stores/modal';
import { usePageTreeTermManager, useSWRxV5MigrationStatus } from '~/stores/page-listing';
import {
  useSWRxSearch,
} from '~/stores/search';
import { useGlobalSocket } from '~/stores/websocket';

import { MenuItemType } from './Common/Dropdown/PageItemControl';
import PaginationWrapper from './PaginationWrapper';
import { PrivateLegacyPagesMigrationModal } from './PrivateLegacyPagesMigrationModal';
import { OperateAllControl } from './SearchPage/OperateAllControl';
import SearchControl from './SearchPage/SearchControl';
import { IReturnSelectedPageIds, SearchPageBase, usePageDeleteModalForBulkDeletion } from './SearchPage2/SearchPageBase';


// TODO: replace with "customize:showPageLimitationS"
const INITIAL_PAGING_SIZE = 20;

const initQ = '/';


/**
 * SearchResultListHead
 */

type SearchResultListHeadProps = {
  searchResult: IFormattedSearchResult,
  offset: number,
  pagingSize: number,
  onPagingSizeChanged: (size: number) => void,
  migrationStatus?: V5MigrationStatus,
}

const SearchResultListHead = React.memo((props: SearchResultListHeadProps): JSX.Element => {
  const { t } = useTranslation();

  const {
    searchResult, offset, pagingSize,
    onPagingSizeChanged, migrationStatus,
  } = props;

  if (migrationStatus == null) {
    return (
      <div className="mw-0 flex-grow-1 flex-basis-0 m-5 text-muted text-center">
        <i className="fa fa-2x fa-spinner fa-pulse mr-1"></i>
      </div>
    );
  }

  const { took, total, hitsCount } = searchResult.meta;
  const leftNum = offset + 1;
  const rightNum = offset + hitsCount;

  const isSuccess = migrationStatus.migratablePagesCount === 0;

  if (isSuccess) {
    return (
      <div className="card border-success mt-3" data-testid="search-result-private-legacy-pages">
        <div className="card-body">
          <h2 className="card-title text-success">{t('private_legacy_pages.nopages_title')}</h2>
          <p className="card-text">
            {t('private_legacy_pages.nopages_desc1')}<br />
            {/* eslint-disable-next-line react/no-danger */}
            <span dangerouslySetInnerHTML={{ __html: t('private_legacy_pages.detail_info') }}></span>
          </p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="form-inline d-flex align-items-center justify-content-between">
        <div className="text-nowrap">
          {t('search_result.result_meta')}
          <span className="ml-3">{`${leftNum}-${rightNum}`} / {total}</span>
          { took != null && (
            <span className="ml-3 text-muted">({took}ms)</span>
          ) }
        </div>
        <div className="input-group flex-nowrap search-result-select-group ml-auto d-md-flex d-none">
          <div className="input-group-prepend">
            <label className="input-group-text text-muted" htmlFor="inputGroupSelect01">{t('search_result.number_of_list_to_display')}</label>
          </div>
          <select
            defaultValue={pagingSize}
            className="custom-select"
            id="inputGroupSelect01"
            onChange={e => onPagingSizeChanged(Number(e.target.value))}
          >
            {[20, 50, 100, 200].map((limit) => {
              return <option key={limit} value={limit}>{limit} {t('search_result.page_number_unit')}</option>;
            })}
          </select>
        </div>
      </div>
      <div className="card border-warning mt-3">
        <div className="card-body">
          <h2 className="card-title text-warning">{t('private_legacy_pages.alert_title')}</h2>
          <p className="card-text">
            {t('private_legacy_pages.alert_desc1', { delete_all_selected_page: t('search_result.delete_all_selected_page') })}<br />
            {/* eslint-disable-next-line react/no-danger */}
            <span dangerouslySetInnerHTML={{ __html: t('private_legacy_pages.detail_info') }}></span>
          </p>
        </div>
      </div>
    </>
  );
});

SearchResultListHead.displayName = 'SearchResultListHead';

/*
 * ConvertByPathModal
 */
type ConvertByPathModalProps = {
  isOpen: boolean,
  close?: () => void,
  onSubmit?: (convertPath: string) => Promise<void> | void,
}
const ConvertByPathModal = React.memo((props: ConvertByPathModalProps): JSX.Element => {
  const { t } = useTranslation();

  const [currentInput, setInput] = useState<string>('');
  const [checked, setChecked] = useState<boolean>(false);

  useEffect(() => {
    setChecked(false);
  }, [props.isOpen]);

  return (
    <Modal size="lg" isOpen={props.isOpen} toggle={props.close} className="grw-create-page">
      <ModalHeader tag="h4" toggle={props.close} className="bg-primary text-light">
        { t('private_legacy_pages.by_path_modal.title') }
      </ModalHeader>
      <ModalBody>
        <p>{t('private_legacy_pages.by_path_modal.description')}</p>
        <input type="text" className="form-control" placeholder="/" value={currentInput} onChange={e => setInput(e.target.value)} />
        <div className="alert alert-danger mt-3" role="alert">
          { t('private_legacy_pages.by_path_modal.alert') }
        </div>
      </ModalBody>
      <ModalFooter>
        <div className="form-check">
          <input
            className="form-check-input"
            type="checkbox"
            id="understoodCheckbox"
            onChange={e => setChecked(e.target.checked)}
          />
          <label className="form-check-label" htmlFor="understoodCheckbox">{ t('private_legacy_pages.by_path_modal.checkbox_label') }</label>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          disabled={!checked}
          onClick={() => props.onSubmit?.(currentInput)}
        >
          <i className="icon-fw icon-refresh" aria-hidden="true"></i>
          { t('private_legacy_pages.by_path_modal.button_label') }
        </button>
      </ModalFooter>
    </Modal>
  );
});

ConvertByPathModal.displayName = 'ConvertByPathModal';

/**
 * LegacyPage
 */

const PrivateLegacyPages = (): JSX.Element => {
  const { t } = useTranslation();
  const { data: currentUser } = useCurrentUser();

  const isAdmin = currentUser?.admin;

  const [keyword, setKeyword] = useState<string>(initQ);
  const [offset, setOffset] = useState<number>(0);
  const [limit, setLimit] = useState<number>(INITIAL_PAGING_SIZE);
  const [isOpenConvertModal, setOpenConvertModal] = useState<boolean>(false);

  const [isControlEnabled, setControlEnabled] = useState(false);

  const selectAllControlRef = useRef<ISelectableAndIndeterminatable|null>(null);
  const searchPageBaseRef = useRef<ISelectableAll & IReturnSelectedPageIds|null>(null);

  const { data, conditions, mutate } = useSWRxSearch(keyword, 'PrivateLegacyPages', {
    offset,
    limit,
    includeUserPages: true,
    includeTrashPages: false,
  });

  const { data: migrationStatus, mutate: mutateMigrationStatus } = useSWRxV5MigrationStatus();
  const { advance: advancePt } = usePageTreeTermManager();

  const searchInvokedHandler = useCallback((_keyword: string) => {
    mutateMigrationStatus();
    setKeyword(_keyword);
    setOffset(0);
  }, []);

  const { open: openModal, close: closeModal } = usePrivateLegacyPagesMigrationModal();
  const { data: socket } = useGlobalSocket();

  useEffect(() => {
    socket?.on(SocketEventName.PageMigrationSuccess, () => {
      toastSuccess(t('private_legacy_pages.toaster.page_migration_succeeded'));
    });

    socket?.on(SocketEventName.PageMigrationError, (data?: PageMigrationErrorData) => {
      if (data == null || data.paths.length === 0) {
        toastError(t('private_legacy_pages.toaster.page_migration_failed'));
      }
      else {
        const errorPaths = data.paths.length > 3
          ? `${data.paths.slice(0, 3).join(', ')}...`
          : data.paths.join(', ');
        toastError(t('private_legacy_pages.toaster.page_migration_failed_with_paths', { paths: errorPaths }));
      }
    });

    return () => {
      socket?.off(SocketEventName.PageMigrationSuccess);
      socket?.off(SocketEventName.PageMigrationError);
    };
  }, [socket]);

  const selectAllCheckboxChangedHandler = useCallback((isChecked: boolean) => {
    const instance = searchPageBaseRef.current;

    if (instance == null) {
      return;
    }

    if (isChecked) {
      instance.selectAll();
      setControlEnabled(true);
    }
    else {
      instance.deselectAll();
      setControlEnabled(false);
    }
  }, []);

  const selectedPagesByCheckboxesChangedHandler = useCallback((selectedCount: number, totalCount: number) => {
    const instance = selectAllControlRef.current;

    if (instance == null) {
      return;
    }

    if (selectedCount === 0) {
      instance.deselect();
      setControlEnabled(false);
    }
    else if (selectedCount === totalCount) {
      instance.select();
      setControlEnabled(true);
    }
    else {
      instance.setIndeterminate();
      setControlEnabled(true);
    }
  }, []);

  // for bulk deletion
  const deleteAllButtonClickedHandler = usePageDeleteModalForBulkDeletion(data, searchPageBaseRef, () => mutate());

  const convertMenuItemClickedHandler = useCallback(() => {
    if (data == null) {
      return;
    }

    const instance = searchPageBaseRef.current;
    if (instance == null || instance.getSelectedPageIds == null) {
      return;
    }

    const selectedPageIds = instance.getSelectedPageIds();

    if (selectedPageIds.size === 0) {
      return;
    }

    const selectedPages = data.data
      .filter(pageWithMeta => selectedPageIds.has(pageWithMeta.data._id))
      .map(pageWithMeta => ({ pageId: pageWithMeta.data._id, path: pageWithMeta.data.path } as ILegacyPrivatePage));

    openModal(
      selectedPages,
      () => {
        toastSuccess(t('Successfully requested'));
        closeModal();
        mutateMigrationStatus();
        mutate();
        advancePt();
      },
    );
  }, [data, mutate, openModal, closeModal, mutateMigrationStatus]);

  const pagingSizeChangedHandler = useCallback((pagingSize: number) => {
    setOffset(0);
    setLimit(pagingSize);
    mutate();
  }, [mutate]);

  const pagingNumberChangedHandler = useCallback((activePage: number) => {
    setOffset((activePage - 1) * limit);
    mutate();
  }, [limit, mutate]);

  const openConvertModalHandler = useCallback(() => {
    if (!isAdmin) { return }
    setOpenConvertModal(true);
  }, [isAdmin]);

  const hitsCount = data?.meta.hitsCount;

  const renderOpenModalButton = useCallback(() => {
    return (
      <div className="d-flex pl-md-2">
        <button type="button" className="btn btn-light" onClick={() => openConvertModalHandler()}>
          {t('private_legacy_pages.input_path_to_convert')}
        </button>
      </div>
    );
  }, [t, openConvertModalHandler]);

  const searchControlAllAction = useMemo(() => {
    const isCheckboxDisabled = hitsCount === 0;

    return (
      <div className="search-control d-flex align-items-center">
        <div className="d-flex pl-md-2">
          <OperateAllControl
            ref={selectAllControlRef}
            isCheckboxDisabled={isCheckboxDisabled}
            onCheckboxChanged={selectAllCheckboxChangedHandler}
          >
            <UncontrolledButtonDropdown>
              <DropdownToggle caret color="outline-primary" disabled={!isControlEnabled}>
                {t('private_legacy_pages.bulk_operation')}
              </DropdownToggle>
              <DropdownMenu>
                <DropdownItem onClick={convertMenuItemClickedHandler}>
                  <i className="icon-fw icon-refresh"></i>
                  {t('private_legacy_pages.convert_all_selected_pages')}
                </DropdownItem>
                <DropdownItem onClick={deleteAllButtonClickedHandler}>
                  <span className="text-danger">
                    <i className="icon-fw icon-trash"></i>
                    {t('search_result.delete_all_selected_page')}
                  </span>
                </DropdownItem>
              </DropdownMenu>
            </UncontrolledButtonDropdown>
          </OperateAllControl>
        </div>
        {isAdmin && renderOpenModalButton()}
      </div>
    );
  }, [convertMenuItemClickedHandler, deleteAllButtonClickedHandler, hitsCount, isControlEnabled, selectAllCheckboxChangedHandler, t]);

  const searchControl = useMemo(() => {
    return (
      <SearchControl
        isSearchServiceReachable
        isEnableSort={false}
        isEnableFilter={false}
        initialSearchConditions={{ keyword: initQ, limit: INITIAL_PAGING_SIZE }}
        onSearchInvoked={searchInvokedHandler}
        allControl={searchControlAllAction}
      />
    );
  }, [searchInvokedHandler, searchControlAllAction]);

  const searchResultListHead = useMemo(() => {
    if (data == null) {
      return <></>;
    }
    return (
      <SearchResultListHead
        searchResult={data}
        offset={offset}
        pagingSize={limit}
        onPagingSizeChanged={pagingSizeChangedHandler}
        migrationStatus={migrationStatus}
      />
    );
  }, [data, limit, offset, pagingSizeChangedHandler, migrationStatus]);

  const searchPager = useMemo(() => {
    // when pager is not needed
    if (data == null || data.meta.hitsCount === data.meta.total) {
      return <></>;
    }

    const { total } = data.meta;
    const { offset, limit } = conditions;

    return (
      <PaginationWrapper
        activePage={Math.floor(offset / limit) + 1}
        totalItemsCount={total}
        pagingLimit={limit}
        changePage={pagingNumberChangedHandler}
      />
    );
  }, [conditions, data, pagingNumberChangedHandler]);

  return (
    <>
      <SearchPageBase
        ref={searchPageBaseRef}
        pages={data?.data}
        onSelectedPagesByCheckboxesChanged={selectedPagesByCheckboxesChangedHandler}
        forceHideMenuItems={[MenuItemType.BOOKMARK, MenuItemType.RENAME, MenuItemType.DUPLICATE, MenuItemType.REVERT, MenuItemType.PATH_RECOVERY]}
        // Components
        searchControl={searchControl}
        searchResultListHead={searchResultListHead}
        searchPager={searchPager}
      />

      <PrivateLegacyPagesMigrationModal />
      <ConvertByPathModal
        isOpen={isOpenConvertModal}
        close={() => setOpenConvertModal(false)}
        onSubmit={async(convertPath: string) => {
          try {
            await apiv3Post<void>('/pages/convert-pages-by-path', {
              convertPath,
            });
            toastSuccess(t('private_legacy_pages.by_path_modal.success'));
            setOpenConvertModal(false);
            mutate();
            advancePt();
          }
          catch (errs) {
            if (errs.length === 1) {
              switch (errs[0].code) {
                case V5ConversionErrCode.GRANT_INVALID:
                  toastError(t('private_legacy_pages.by_path_modal.error_grant_invalid'));
                  break;
                case V5ConversionErrCode.PAGE_NOT_FOUND:
                  toastError(t('private_legacy_pages.by_path_modal.error_page_not_found'));
                  break;
                case V5ConversionErrCode.DUPLICATE_PAGES_FOUND:
                  toastError(t('private_legacy_pages.by_path_modal.error_duplicate_pages_found'));
                  break;
                default:
                  toastError(t('private_legacy_pages.by_path_modal.error'));
              }
            }
            else {
              toastError(t('private_legacy_pages.by_path_modal.error'));
            }
          }
        }}
      />
    </>
  );
};

export default PrivateLegacyPages;
