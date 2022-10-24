import React, {
  useCallback, useMemo, useState,
} from 'react';

import { HasObjectId, IAttachment } from '@growi/core';

import { useSWRxAttachments } from '~/stores/attachment';
import { useEditingMarkdown, useCurrentPageId, useIsGuestUser } from '~/stores/context';

import { DeleteAttachmentModal } from './PageAttachment/DeleteAttachmentModal';
import { PageAttachmentList } from './PageAttachment/PageAttachmentList';
import PaginationWrapper from './PaginationWrapper';

// Utility
const checkIfFileInUse = (markdown: string, attachment): boolean => {
  return markdown.indexOf(attachment._id) >= 0;
};

const PageAttachment = (): JSX.Element => {
  // Static SWRs
  const { data: pageId } = useCurrentPageId();
  const { data: isGuestUser } = useIsGuestUser();
  const { data: markdown } = useEditingMarkdown();

  // States
  const [pageNumber, setPageNumber] = useState(1);
  const [attachmentToDelete, setAttachmentToDelete] = useState<(IAttachment & HasObjectId) | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState('');

  // SWRs
  const { data: dataAttachments, remove } = useSWRxAttachments(pageId, pageNumber);

  // Custom hooks
  const inUseAttachmentsMap: { [id: string]: boolean } | undefined = useMemo(() => {
    if (markdown == null || dataAttachments == null) {
      return undefined;
    }

    const attachmentEntries = dataAttachments.attachments
      .map((attachment) => {
        return [attachment._id, checkIfFileInUse(markdown, attachment)];
      });

    return Object.fromEntries(attachmentEntries);
  }, [dataAttachments, markdown]);

  // Methods
  const onChangePageHandler = useCallback((newPageNumber: number) => {
    setPageNumber(newPageNumber);
  }, []);

  const onAttachmentDeleteClicked = useCallback((attachment) => {
    setAttachmentToDelete(attachment);
  }, []);

  const onAttachmentDeleteClickedConfirmHandler = useCallback(async(attachment: IAttachment & HasObjectId) => {
    setDeleting(true);

    try {
      await remove({ attachment_id: attachment._id });

      setAttachmentToDelete(null);
      setDeleting(false);
    }
    catch {
      setDeleteError('Something went wrong.');
      setDeleting(false);
    }
  }, [remove]);

  const onToggleHandler = useCallback(() => {
    setAttachmentToDelete(null);
    setDeleteError('');
  }, []);

  // Renderers
  const renderPageAttachmentList = useCallback(() => {
    if (dataAttachments == null || inUseAttachmentsMap == null) {
      return (
        <div className="text-muted text-center">
          <i className="fa fa-2x fa-spinner fa-pulse mr-1"></i>
        </div>
      );
    }

    return (
      <PageAttachmentList
        attachments={dataAttachments.attachments}
        inUse={inUseAttachmentsMap}
        onAttachmentDeleteClicked={onAttachmentDeleteClicked}
        isUserLoggedIn={!isGuestUser}
      />
    );
  }, [dataAttachments, inUseAttachmentsMap, isGuestUser, onAttachmentDeleteClicked]);

  const renderDeleteAttachmentModal = useCallback(() => {
    if (isGuestUser) {
      return <></>;
    }

    if (dataAttachments == null || dataAttachments.attachments.length === 0 || attachmentToDelete == null) {
      return <></>;
    }

    const isOpen = attachmentToDelete != null;

    return (
      <DeleteAttachmentModal
        isOpen={isOpen}
        toggle={onToggleHandler}
        attachmentToDelete={attachmentToDelete}
        deleting={deleting}
        deleteError={deleteError}
        onAttachmentDeleteClickedConfirm={onAttachmentDeleteClickedConfirmHandler}
      />
    );
  // eslint-disable-next-line max-len
  }, [attachmentToDelete, dataAttachments, deleteError, deleting, isGuestUser, onAttachmentDeleteClickedConfirmHandler, onToggleHandler]);

  const renderPaginationWrapper = useCallback(() => {
    if (dataAttachments == null || dataAttachments.attachments.length === 0) {
      return <></>;
    }

    return (
      <PaginationWrapper
        activePage={pageNumber}
        changePage={onChangePageHandler}
        totalItemsCount={dataAttachments.totalAttachments}
        pagingLimit={dataAttachments.limit}
        align="center"
      />
    );
  }, [dataAttachments, onChangePageHandler, pageNumber]);

  return (
    <div data-testid="page-attachment">
      {renderPageAttachmentList()}

      {renderDeleteAttachmentModal()}

      {renderPaginationWrapper()}
    </div>
  );
};

export default PageAttachment;
