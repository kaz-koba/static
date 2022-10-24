import React, {
  useEffect, useState, useMemo,
} from 'react';

import { pagePathUtils, pathUtils } from '@growi/core';
import { format } from 'date-fns';
import { useTranslation } from 'next-i18next';
import { useRouter } from 'next/router';
import { Modal, ModalHeader, ModalBody } from 'reactstrap';
import { debounce } from 'throttle-debounce';


import { toastError } from '~/client/util/apiNotification';
import { useCurrentUser, useIsSearchServiceReachable } from '~/stores/context';
import { usePageCreateModal } from '~/stores/modal';
import { EditorMode, useEditorMode } from '~/stores/ui';

import PagePathAutoComplete from './PagePathAutoComplete';

const {
  userPageRoot, isCreatablePage, generateEditorPath, isUsersHomePage,
} = pagePathUtils;

const PageCreateModal = () => {
  const { t } = useTranslation();
  const router = useRouter();

  const { data: currentUser } = useCurrentUser();

  const { data: pageCreateModalData, close: closeCreateModal } = usePageCreateModal();
  const { isOpened, path } = pageCreateModalData;

  const { data: isReachable } = useIsSearchServiceReachable();
  const pathname = path || '';
  const userPageRootPath = userPageRoot(currentUser);
  const isCreatable = isCreatablePage(pathname) || isUsersHomePage(pathname);
  const pageNameInputInitialValue = isCreatable ? pathUtils.addTrailingSlash(pathname) : '/';
  const now = format(new Date(), 'yyyy/MM/dd');

  const { mutate: mutateEditorMode } = useEditorMode();

  const [todayInput1, setTodayInput1] = useState(t('Memo'));
  const [todayInput2, setTodayInput2] = useState('');
  const [pageNameInput, setPageNameInput] = useState(pageNameInputInitialValue);
  const [template, setTemplate] = useState(null);
  const [isMatchedWithUserHomePagePath, setIsMatchedWithUserHomePagePath] = useState(false);

  // ensure pageNameInput is synced with selectedPagePath || currentPagePath
  useEffect(() => {
    if (isOpened) {
      setPageNameInput(isCreatable ? pathUtils.addTrailingSlash(pathname) : '/');
    }
  }, [isOpened, pathname, isCreatable]);

  const checkIsUsersHomePageDebounce = useMemo(() => {
    const checkIsUsersHomePage = () => {
      setIsMatchedWithUserHomePagePath(isUsersHomePage(pageNameInput));
    };

    return debounce(1000, checkIsUsersHomePage);
  }, [pageNameInput]);

  useEffect(() => {
    if (isOpened) {
      checkIsUsersHomePageDebounce(pageNameInput);
    }
  }, [isOpened, checkIsUsersHomePageDebounce, pageNameInput]);

  function transitBySubmitEvent(e, transitHandler) {
    // prevent page transition by submit
    e.preventDefault();
    transitHandler();
  }

  /**
   * change todayInput1
   * @param {string} value
   */
  function onChangeTodayInput1Handler(value) {
    setTodayInput1(value);
  }

  /**
   * change todayInput2
   * @param {string} value
   */
  function onChangeTodayInput2Handler(value) {
    setTodayInput2(value);
  }

  /**
   * change template
   * @param {string} value
   */
  function onChangeTemplateHandler(value) {
    setTemplate(value);
  }

  /**
   * join path, check if creatable, then redirect
   * @param {string} paths
   */
  async function redirectToEditor(...paths) {
    try {
      const editorPath = generateEditorPath(...paths);
      await router.push(editorPath);
      mutateEditorMode(EditorMode.Editor);

      // close modal
      closeCreateModal();
    }
    catch (err) {
      toastError(err);
    }
  }

  /**
   * access today page
   */
  function createTodayPage() {
    let tmpTodayInput1 = todayInput1;
    if (tmpTodayInput1 === '') {
      tmpTodayInput1 = t('Memo');
    }
    redirectToEditor(userPageRootPath, tmpTodayInput1, now, todayInput2);
  }

  /**
   * access input page
   */
  function createInputPage() {
    redirectToEditor(pageNameInput);
  }

  function ppacSubmitHandler(input) {
    redirectToEditor(input);
  }

  /**
   * access template page
   */
  function createTemplatePage(e) {
    const pageName = (template === 'children') ? '_template' : '__template';
    redirectToEditor(pathname, pageName);
  }

  function renderCreateTodayForm() {
    if (!isOpened) {
      return <></>;
    }
    return (
      <div className="row">
        <fieldset className="col-12 mb-4">
          <h3 className="grw-modal-head pb-2">{t("Create today's")}</h3>

          <div className="d-sm-flex align-items-center justify-items-between">

            <div className="d-flex align-items-center flex-fill flex-wrap flex-lg-nowrap">
              <div className="d-flex align-items-center">
                <span>{userPageRootPath}/</span>
                <form onSubmit={e => transitBySubmitEvent(e, createTodayPage)}>
                  <input
                    type="text"
                    className="page-today-input1 form-control text-center mx-2"
                    value={todayInput1}
                    onChange={e => onChangeTodayInput1Handler(e.target.value)}
                  />
                </form>
                <span className="page-today-suffix">/{now}/</span>
              </div>
              <form className="mt-1 mt-lg-0 ml-lg-2 w-100" onSubmit={e => transitBySubmitEvent(e, createTodayPage)}>
                <input
                  type="text"
                  className="page-today-input2 form-control w-100"
                  id="page-today-input2"
                  placeholder={t('Input page name (optional)')}
                  value={todayInput2}
                  onChange={e => onChangeTodayInput2Handler(e.target.value)}
                />
              </form>
            </div>

            <div className="d-flex justify-content-end mt-1 mt-sm-0">
              <button
                type="button"
                data-testid="btn-create-memo"
                className="grw-btn-create-page btn btn-outline-primary rounded-pill text-nowrap ml-3"
                onClick={createTodayPage}
              >
                <i className="icon-fw icon-doc"></i>{t('Create')}
              </button>
            </div>

          </div>

        </fieldset>
      </div>
    );
  }

  function renderInputPageForm() {
    if (!isOpened) {
      return <></>;
    }
    return (
      <div className="row" data-testid="row-create-page-under-below">
        <fieldset className="col-12 mb-4">
          <h3 className="grw-modal-head pb-2">{t('Create under')}</h3>

          <div className="d-sm-flex align-items-center justify-items-between">
            <div className="flex-fill">
              {isReachable
                ? (
                  <PagePathAutoComplete
                    initializedPath={pageNameInputInitialValue}
                    addTrailingSlash
                    onSubmit={ppacSubmitHandler}
                    onInputChange={value => setPageNameInput(value)}
                    autoFocus
                  />
                )
                : (
                  <form onSubmit={e => transitBySubmitEvent(e, createInputPage)}>
                    <input
                      type="text"
                      value={pageNameInput}
                      className="form-control flex-fill"
                      placeholder={t('Input page name')}
                      onChange={e => setPageNameInput(e.target.value)}
                      required
                    />
                  </form>
                )}
            </div>

            <div className="d-flex justify-content-end mt-1 mt-sm-0">
              <button
                type="button"
                data-testid="btn-create-page-under-below"
                className="grw-btn-create-page btn btn-outline-primary rounded-pill text-nowrap ml-3"
                onClick={createInputPage}
                disabled={isMatchedWithUserHomePagePath}
              >
                <i className="icon-fw icon-doc"></i>{t('Create')}
              </button>
            </div>

          </div>
          { isMatchedWithUserHomePagePath && (
            <p className="text-danger mt-2">Error: Cannot create page under /user page directory.</p>
          ) }

        </fieldset>
      </div>
    );
  }

  function renderTemplatePageForm() {
    if (!isOpened) {
      return <></>;
    }
    return (
      <div className="row">
        <fieldset className="col-12">

          <h3 className="grw-modal-head pb-2">
            {t('template.modal_label.Create template under')}<br />
            <code className="h6">{pathname}</code>
          </h3>

          <div className="d-sm-flex align-items-center justify-items-between">

            <div id="dd-template-type" className="dropdown flex-fill">
              <button id="template-type" type="button" className="btn btn-secondary btn dropdown-toggle w-100" data-toggle="dropdown">
                {template == null && t('template.option_label.select')}
                {template === 'children' && t('template.children.label')}
                {template === 'decendants' && t('template.decendants.label')}
              </button>
              <div className="dropdown-menu" aria-labelledby="userMenu">
                <button className="dropdown-item" type="button" onClick={() => onChangeTemplateHandler('children')}>
                  {t('template.children.label')} (_template)<br className="d-block d-md-none" />
                  <small className="text-muted text-wrap">- {t('template.children.desc')}</small>
                </button>
                <button className="dropdown-item" type="button" onClick={() => onChangeTemplateHandler('decendants')}>
                  {t('template.decendants.label')} (__template) <br className="d-block d-md-none" />
                  <small className="text-muted">- {t('template.decendants.desc')}</small>
                </button>
              </div>
            </div>

            <div className="d-flex justify-content-end mt-1 mt-sm-0">
              <button
                type="button"
                className={`grw-btn-create-page btn btn-outline-primary rounded-pill text-nowrap ml-3 ${template == null && 'disabled'}`}
                onClick={createTemplatePage}
              >
                <i className="icon-fw icon-doc"></i>{t('Edit')}
              </button>
            </div>

          </div>

        </fieldset>
      </div>
    );
  }

  return (
    <Modal
      size="lg"
      isOpen={isOpened}
      toggle={() => closeCreateModal()}
      data-testid="page-create-modal"
      className="grw-create-page"
      autoFocus={false}
    >
      <ModalHeader tag="h4" toggle={() => closeCreateModal()} className="bg-primary text-light">
        {t('New Page')}
      </ModalHeader>
      <ModalBody>
        {renderCreateTodayForm()}
        {renderInputPageForm()}
        {renderTemplatePageForm()}
      </ModalBody>
    </Modal>

  );
};


export default PageCreateModal;
