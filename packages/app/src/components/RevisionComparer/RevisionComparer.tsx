import React, { useState } from 'react';

import { IRevisionHasPageId, pagePathUtils } from '@growi/core';
import { useTranslation } from 'next-i18next';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import {
  Dropdown, DropdownToggle, DropdownMenu, DropdownItem,
} from 'reactstrap';

import { useCurrentPagePath } from '~/stores/context';

import { RevisionDiff } from '../PageHistory/RevisionDiff';

import styles from './RevisionComparer.module.scss';

const { encodeSpaces } = pagePathUtils;

const DropdownItemContents = ({ title, contents }) => (
  <>
    <div className="h6 mt-1 mb-2"><strong>{title}</strong></div>
    <div className="card well mb-1 p-2">{contents}</div>
  </>
);

type RevisionComparerProps = {
  sourceRevision: IRevisionHasPageId
  targetRevision: IRevisionHasPageId
  currentPageId?: string
}

export const RevisionComparer = (props: RevisionComparerProps): JSX.Element => {
  const { t } = useTranslation();

  const {
    sourceRevision, targetRevision, currentPageId,
  } = props;

  const { data: currentPagePath } = useCurrentPagePath();
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };

  const generateURL = (pathName: string) => {
    const { origin } = window.location;

    const url = new URL(pathName, origin);

    if (sourceRevision != null && targetRevision != null) {
      const urlParams = `${sourceRevision._id}...${targetRevision._id}`;
      url.searchParams.set('compare', urlParams);
    }

    return encodeSpaces(decodeURI(url.href));
  };

  const isNodiff = (sourceRevision == null || targetRevision == null) ? true : sourceRevision._id === targetRevision._id;

  if (currentPageId == null || currentPagePath == null) {
    return <>{ t('not_found_page.page_not_exist')}</>;
  }

  return (
    <div className={`${styles['revision-compare']} revision-compare`}>
      <div className="d-flex">
        <h4 className="align-self-center">{ t('page_history.comparing_revisions') }</h4>
        <Dropdown
          className="grw-copy-dropdown align-self-center ml-auto"
          isOpen={dropdownOpen}
          toggle={() => toggleDropdown()}
        >
          <DropdownToggle
            caret
            className="d-block text-muted bg-transparent btn-copy border-0 py-0"
          >
            <i className="ti ti-clipboard"></i>
          </DropdownToggle>
          <DropdownMenu positionFixed right modifiers={{ preventOverflow: { boundariesElement: undefined } }}>
            {/* Page path URL */}
            <CopyToClipboard text={generateURL(currentPagePath)}>
              <DropdownItem className="px-3">
                <DropdownItemContents title={t('copy_to_clipboard.Page URL')} contents={generateURL(currentPagePath)} />
              </DropdownItem>
            </CopyToClipboard>
            {/* Permanent Link URL */}
            <CopyToClipboard text={generateURL(currentPageId)}>
              <DropdownItem className="px-3">
                <DropdownItemContents title={t('copy_to_clipboard.Permanent link')} contents={generateURL(currentPageId)} />
              </DropdownItem>
            </CopyToClipboard>
            <DropdownItem divider className="my-0"></DropdownItem>
          </DropdownMenu>
        </Dropdown>
      </div>

      <div className={`revision-compare-container ${isNodiff ? 'nodiff' : ''}`}>
        { isNodiff
          ? (
            <span className="h3 text-muted">{t('No diff')}</span>
          )
          : (
            <RevisionDiff
              revisionDiffOpened
              previousRevision={sourceRevision}
              currentRevision={targetRevision}
            />
          )
        }
      </div>
    </div>
  );
};
