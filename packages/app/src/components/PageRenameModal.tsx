import React, {
  useState, useEffect, useCallback, useMemo,
} from 'react';

import { pagePathUtils } from '@growi/core';
import { useTranslation } from 'next-i18next';
import {
  Collapse, Modal, ModalHeader, ModalBody, ModalFooter,
} from 'reactstrap';
import { debounce } from 'throttle-debounce';

import { toastError } from '~/client/util/apiNotification';
import { apiv3Get, apiv3Put } from '~/client/util/apiv3-client';
import { isIPageInfoForEntity } from '~/interfaces/page';
import { useSiteUrl, useIsSearchServiceReachable } from '~/stores/context';
import { usePageRenameModal } from '~/stores/modal';
import { useSWRxPageInfo } from '~/stores/page';

import DuplicatedPathsTable from './DuplicatedPathsTable';
import ApiErrorMessageList from './PageManagement/ApiErrorMessageList';
import PagePathAutoComplete from './PagePathAutoComplete';


const isV5Compatible = (meta: unknown): boolean => {
  return isIPageInfoForEntity(meta) ? meta.isV5Compatible : true;
};


const PageRenameModal = (): JSX.Element => {
  const { t } = useTranslation();

  const { isUsersHomePage } = pagePathUtils;
  const { data: siteUrl } = useSiteUrl();
  const { data: renameModalData, close: closeRenameModal } = usePageRenameModal();
  const { data: isReachable } = useIsSearchServiceReachable();

  const isOpened = renameModalData?.isOpened ?? false;
  const page = renameModalData?.page;

  const shouldFetch = isOpened && page != null && !isIPageInfoForEntity(page.meta);
  const { data: pageInfo } = useSWRxPageInfo(shouldFetch ? page?.data._id : null);

  if (page != null && pageInfo != null) {
    page.meta = pageInfo;
  }

  const [pageNameInput, setPageNameInput] = useState('');

  const [errs, setErrs] = useState(null);

  const [subordinatedPages, setSubordinatedPages] = useState([]);
  const [existingPaths, setExistingPaths] = useState<string[]>([]);
  const [canRename, setCanRename] = useState(false);
  const [isRenameRecursively, setIsRenameRecursively] = useState(true);
  const [isRenameRedirect, setIsRenameRedirect] = useState(false);
  const [isRemainMetadata, setIsRemainMetadata] = useState(false);
  const [expandOtherOptions, setExpandOtherOptions] = useState(false);
  const [subordinatedError] = useState(null);
  const [isMatchedWithUserHomePagePath, setIsMatchedWithUserHomePagePath] = useState(false);

  const updateSubordinatedList = useCallback(async() => {
    if (page == null) {
      return;
    }

    const { path } = page.data;
    try {
      const res = await apiv3Get('/pages/subordinated-list', { path });
      setSubordinatedPages(res.data.subordinatedPages);
    }
    catch (err) {
      setErrs(err);
      toastError(t('modal_rename.label.Failed to get subordinated pages'));
    }
  }, [page, t]);

  useEffect(() => {
    if (page != null && isOpened) {
      updateSubordinatedList();
      setPageNameInput(page.data.path);
    }
  }, [isOpened, page, updateSubordinatedList]);

  const rename = useCallback(async() => {
    if (page == null || !canRename) {
      return;
    }

    const _isV5Compatible = isV5Compatible(page.meta);

    setErrs(null);

    const { _id, path, revision } = page.data;
    try {
      const response = await apiv3Put('/pages/rename', {
        pageId: _id,
        revisionId: revision ?? null,
        isRecursively: !_isV5Compatible ? isRenameRecursively : undefined,
        isRenameRedirect,
        updateMetadata: !isRemainMetadata,
        newPagePath: pageNameInput,
        path,
      });

      const { page } = response.data;
      const url = new URL(page.path, 'https://dummy');
      if (isRenameRedirect) {
        url.searchParams.append('withRedirect', 'true');
      }

      const onRenamed = renameModalData?.opts?.onRenamed;
      if (onRenamed != null) {
        onRenamed(path);
      }
      closeRenameModal();
    }
    catch (err) {
      setErrs(err);
    }
  }, [closeRenameModal, canRename, isRemainMetadata, isRenameRecursively, isRenameRedirect, page, pageNameInput, renameModalData?.opts?.onRenamed]);

  const checkExistPaths = useCallback(async(fromPath, toPath) => {
    if (page == null) {
      return;
    }

    try {
      const res = await apiv3Get<{ existPaths: string[]}>('/page/exist-paths', { fromPath, toPath });
      const { existPaths } = res.data;
      if (existPaths.length === 0) {
        setCanRename(true);
      }
      setExistingPaths(existPaths);
    }
    catch (err) {
      setErrs(err);
      toastError(t('modal_rename.label.Failed to get exist path'));
    }
  }, [page, t]);

  const checkExistPathsDebounce = useMemo(() => {
    return debounce(1000, checkExistPaths);
  }, [checkExistPaths]);

  const checkIsUsersHomePageDebounce = useMemo(() => {
    const checkIsPagePathRenameable = () => {
      setIsMatchedWithUserHomePagePath(isUsersHomePage(pageNameInput));
    };

    return debounce(1000, checkIsPagePathRenameable);
  }, [isUsersHomePage, pageNameInput]);

  useEffect(() => {
    if (isOpened && page != null && pageNameInput !== page.data.path) {
      checkExistPathsDebounce(page.data.path, pageNameInput);
      checkIsUsersHomePageDebounce(pageNameInput);
    }
  }, [isOpened, pageNameInput, subordinatedPages, checkExistPathsDebounce, page, checkIsUsersHomePageDebounce]);

  useEffect(() => {
    if (isOpened && page != null) {
      setCanRename(false);
    }
  }, [isOpened, page, pageNameInput]);


  function ppacInputChangeHandler(value) {
    setErrs(null);
    setPageNameInput(value);
  }

  /**
   * change pageNameInput
   * @param {string} value
   */
  function inputChangeHandler(value) {
    setErrs(null);
    setPageNameInput(value);
  }

  useEffect(() => {
    if (isOpened || page == null) {
      return;
    }

    // reset states after the modal closed
    setTimeout(() => {
      setPageNameInput('');
      setErrs(null);
      setSubordinatedPages([]);
      setExistingPaths([]);
      setIsRenameRecursively(true);
      setIsRenameRedirect(false);
      setIsRemainMetadata(false);
      setExpandOtherOptions(false);
    }, 1000);

  }, [isOpened, page]);

  const bodyContent = () => {
    if (!isOpened || page == null) {
      return <></>;
    }

    const { path } = page.data;
    const isTargetPageDuplicate = existingPaths.includes(pageNameInput);

    return (
      <>
        <div className="form-group">
          <label>{ t('modal_rename.label.Current page name') }</label><br />
          <code>{ path }</code>
        </div>
        <div className="form-group">
          <label htmlFor="newPageName">{ t('modal_rename.label.New page name') }</label><br />
          <div className="input-group">
            <div className="input-group-prepend">
              <span className="input-group-text">{siteUrl}</span>
            </div>
            <form className="flex-fill" onSubmit={(e) => { e.preventDefault(); rename() }}>
              {isReachable
                ? (
                  <PagePathAutoComplete
                    initializedPath={path}
                    onSubmit={rename}
                    onInputChange={ppacInputChangeHandler}
                    autoFocus
                  />
                )
                : (
                  <input
                    type="text"
                    value={pageNameInput}
                    className="form-control"
                    onChange={e => inputChangeHandler(e.target.value)}
                    required
                    autoFocus
                  />
                )}
            </form>
          </div>
        </div>

        { isTargetPageDuplicate && (
          <p className="text-danger">Error: Target path is duplicated.</p>
        ) }
        { isMatchedWithUserHomePagePath && (
          <p className="text-danger">Error: Cannot move to directory under /user page.</p>
        ) }

        { !isV5Compatible(page.meta) && (
          <>
            <div className="custom-control custom-radio custom-radio-warning">
              <input
                className="custom-control-input"
                name="recursively"
                id="cbRenameThisPageOnly"
                type="radio"
                checked={!isRenameRecursively}
                onChange={() => setIsRenameRecursively(!isRenameRecursively)}
              />
              <label className="custom-control-label" htmlFor="cbRenameThisPageOnly">
                { t('modal_rename.label.Rename this page only') }
              </label>
            </div>
            <div className="custom-control custom-radio custom-radio-warning mt-1">
              <input
                className="custom-control-input"
                name="withoutExistRecursively"
                id="cbForceRenameRecursively"
                type="radio"
                checked={isRenameRecursively}
                onChange={() => setIsRenameRecursively(!isRenameRecursively)}
              />
              <label className="custom-control-label" htmlFor="cbForceRenameRecursively">
                { t('modal_rename.label.Force rename all child pages') }
                <p className="form-text text-muted mt-0">{ t('modal_rename.help.recursive') }</p>
              </label>
              {isRenameRecursively && existingPaths.length !== 0 && (
                <DuplicatedPathsTable existingPaths={existingPaths} fromPath={path} toPath={pageNameInput} />
              ) }
            </div>
          </>
        ) }

        <p className="mt-2">
          <button type="button" className="btn btn-link mt-2 p-0" aria-expanded="false" onClick={() => setExpandOtherOptions(!expandOtherOptions)}>
            <i className={`fa fa-fw fa-arrow-right ${expandOtherOptions ? 'fa-rotate-90' : ''}`}></i>
            { t('modal_rename.label.Other options') }
          </button>
        </p>
        <Collapse isOpen={expandOtherOptions}>
          <div className="custom-control custom-checkbox custom-checkbox-success">
            <input
              className="custom-control-input"
              name="create_redirect"
              id="cbRenameRedirect"
              type="checkbox"
              checked={isRenameRedirect}
              onChange={() => setIsRenameRedirect(!isRenameRedirect)}
            />
            <label className="custom-control-label" htmlFor="cbRenameRedirect">
              { t('modal_rename.label.Redirect') }
              <p className="form-text text-muted mt-0">{ t('modal_rename.help.redirect') }</p>
            </label>
          </div>

          <div className="custom-control custom-checkbox custom-checkbox-primary">
            <input
              className="custom-control-input"
              name="remain_metadata"
              id="cbRemainMetadata"
              type="checkbox"
              checked={isRemainMetadata}
              onChange={() => setIsRemainMetadata(!isRemainMetadata)}
            />
            <label className="custom-control-label" htmlFor="cbRemainMetadata">
              { t('modal_rename.label.Do not update metadata') }
              <p className="form-text text-muted mt-0">{ t('modal_rename.help.metadata') }</p>
            </label>
          </div>
          <div> {subordinatedError} </div>
        </Collapse>
      </>
    );
  };

  const footerContent = () => {
    if (!isOpened || page == null) {
      return <></>;
    }

    let submitButtonDisabled = false;

    if (isMatchedWithUserHomePagePath) {
      submitButtonDisabled = true;
    }
    else if (!canRename) {
      submitButtonDisabled = true;
    }
    else if (isV5Compatible(page.meta)) {
      submitButtonDisabled = existingPaths.length !== 0; // v5 data
    }
    else {
      submitButtonDisabled = !isRenameRecursively; // v4 data
    }
    return (
      <>
        <ApiErrorMessageList errs={errs} targetPath={pageNameInput} />
        <button
          type="button"
          className="btn btn-primary"
          onClick={rename}
          disabled={submitButtonDisabled}
        >Rename
        </button>
      </>
    );
  };


  return (
    <Modal size="lg" isOpen={isOpened} toggle={closeRenameModal} data-testid="page-rename-modal" autoFocus={false}>
      <ModalHeader tag="h4" toggle={closeRenameModal} className="bg-primary text-light">
        { t('modal_rename.label.Move/Rename page') }
      </ModalHeader>
      <ModalBody>
        {bodyContent()}
      </ModalBody>
      <ModalFooter>
        {footerContent()}
      </ModalFooter>
    </Modal>
  );
};

export default PageRenameModal;
