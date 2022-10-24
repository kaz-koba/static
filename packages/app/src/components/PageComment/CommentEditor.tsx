import React, {
  useCallback, useState, useRef, useEffect,
} from 'react';

import { UserPicture } from '@growi/ui';
import dynamic from 'next/dynamic';
import {
  Button, TabContent, TabPane,
} from 'reactstrap';
import * as toastr from 'toastr';

import { apiPostForm } from '~/client/util/apiv1-client';
import { IEditorMethods } from '~/interfaces/editor-methods';
import { useSWRxPageComment } from '~/stores/comment';
import {
  useCurrentPagePath, useCurrentUser, useRevisionId, useIsSlackConfigured,
  useIsUploadableFile, useIsUploadableImage,
} from '~/stores/context';
import { useSWRxSlackChannels, useIsSlackEnabled } from '~/stores/editor';

import { CustomNavTab } from '../CustomNavigation/CustomNav';
import NotAvailableForGuest from '../NotAvailableForGuest';
import Editor from '../PageEditor/Editor';


import { CommentPreview } from './CommentPreview';

import styles from './CommentEditor.module.scss';


const SlackNotification = dynamic(() => import('../SlackNotification').then(mod => mod.SlackNotification), { ssr: false });


const navTabMapping = {
  comment_editor: {
    Icon: () => <i className="icon-settings" />,
    i18n: 'Write',
    index: 0,
  },
  comment_preview: {
    Icon: () => <i className="icon-settings" />,
    i18n: 'Preview',
    index: 1,
  },
};

export type CommentEditorProps = {
  pageId: string,
  isForNewComment?: boolean,
  replyTo?: string,
  currentCommentId?: string,
  commentBody?: string,
  onCancelButtonClicked?: () => void,
  onCommentButtonClicked?: () => void,
}


export const CommentEditor = (props: CommentEditorProps): JSX.Element => {

  const {
    pageId, isForNewComment, replyTo,
    currentCommentId, commentBody, onCancelButtonClicked, onCommentButtonClicked,
  } = props;

  const { data: currentUser } = useCurrentUser();
  const { data: currentPagePath } = useCurrentPagePath();
  const { update: updateComment, post: postComment } = useSWRxPageComment(pageId);
  const { data: revisionId } = useRevisionId();
  const { data: isSlackEnabled, mutate: mutateIsSlackEnabled } = useIsSlackEnabled();
  const { data: slackChannelsData } = useSWRxSlackChannels(currentPagePath);
  const { data: isSlackConfigured } = useIsSlackConfigured();
  const { data: isUploadableFile } = useIsUploadableFile();
  const { data: isUploadableImage } = useIsUploadableImage();

  const [isReadyToUse, setIsReadyToUse] = useState(!isForNewComment);
  const [comment, setComment] = useState(commentBody ?? '');
  const [activeTab, setActiveTab] = useState('comment_editor');
  const [error, setError] = useState();
  const [slackChannels, setSlackChannels] = useState<string>('');

  const editorRef = useRef<IEditorMethods>(null);

  const handleSelect = useCallback((activeTab: string) => {
    setActiveTab(activeTab);
  }, []);

  useEffect(() => {
    if (slackChannelsData != null) {
      setSlackChannels(slackChannelsData.toString());
      mutateIsSlackEnabled(false);
    }
  }, [mutateIsSlackEnabled, slackChannelsData]);

  const isSlackEnabledToggleHandler = (isSlackEnabled: boolean) => {
    mutateIsSlackEnabled(isSlackEnabled, false);
  };

  const slackChannelsChangedHandler = useCallback((slackChannels: string) => {
    setSlackChannels(slackChannels);
  }, []);

  const initializeEditor = useCallback(() => {
    setComment('');
    setActiveTab('comment_editor');
    setError(undefined);
    // reset value
    if (editorRef.current == null) { return }
    editorRef.current.setValue('');
  }, []);

  const cancelButtonClickedHandler = useCallback(() => {
    // change state to not ready
    // when this editor is for the new comment mode
    if (isForNewComment) {
      setIsReadyToUse(false);
    }

    if (onCancelButtonClicked != null) {
      onCancelButtonClicked();
    }
  }, [isForNewComment, onCancelButtonClicked]);

  const postCommentHandler = useCallback(async() => {
    try {
      if (currentCommentId != null) {
        // update current comment
        await updateComment(comment, revisionId, currentCommentId);
      }
      else {
        // post new comment
        const postCommentArgs = {
          commentForm: {
            comment,
            revisionId,
            replyTo,
          },
          slackNotificationForm: {
            isSlackEnabled,
            slackChannels,
          },
        };
        await postComment(postCommentArgs);
      }

      initializeEditor();

      if (onCommentButtonClicked != null) {
        onCommentButtonClicked();
      }
    }
    catch (err) {
      const errorMessage = err.message || 'An unknown error occured when posting comment';
      setError(errorMessage);
    }
  }, [
    comment, currentCommentId, initializeEditor,
    isSlackEnabled, onCommentButtonClicked, replyTo, slackChannels,
    postComment, revisionId, updateComment,
  ]);

  const ctrlEnterHandler = useCallback((event) => {
    if (event != null) {
      event.preventDefault();
    }

    postCommentHandler();
  }, [postCommentHandler]);

  const apiErrorHandler = useCallback((error: Error) => {
    toastr.error(error.message, 'Error occured', {
      closeButton: true,
      progressBar: true,
      newestOnTop: false,
      showDuration: '100',
      hideDuration: '100',
      timeOut: '3000',
    });
  }, []);

  const uploadHandler = useCallback(async(file) => {
    if (editorRef.current == null) { return }

    const pagePath = currentPagePath;
    const endpoint = '/attachments.add';
    const formData = new FormData();
    formData.append('file', file);
    formData.append('path', pagePath ?? '');
    formData.append('page_id', pageId ?? '');

    try {
      // TODO: typescriptize res
      const res = await apiPostForm(endpoint, formData) as any;
      const attachment = res.attachment;
      const fileName = attachment.originalName;
      let insertText = `[${fileName}](${attachment.filePathProxied})`;
      // when image
      if (attachment.fileFormat.startsWith('image/')) {
        // modify to "![fileName](url)" syntax
        insertText = `!${insertText}`;
      }
      editorRef.current.insertText(insertText);
    }
    catch (err) {
      apiErrorHandler(err);
    }
    finally {
      editorRef.current.terminateUploadingState();
    }
  }, [apiErrorHandler, currentPagePath, pageId]);

  const getCommentHtml = useCallback(() => {
    if (currentPagePath == null) {
      return <></>;
    }

    return <CommentPreview markdown={comment} />;
  }, [currentPagePath, comment]);

  const renderBeforeReady = useCallback((): JSX.Element => {
    return (
      <div className="text-center">
        <NotAvailableForGuest>
          <button
            type="button"
            className="btn btn-lg btn-link"
            onClick={() => setIsReadyToUse(true)}
          >
            <i className="icon-bubble"></i> Add Comment
          </button>
        </NotAvailableForGuest>
      </div>
    );
  }, []);

  const renderReady = () => {
    const commentPreview = getCommentHtml();

    const errorMessage = <span className="text-danger text-right mr-2">{error}</span>;
    const cancelButton = (
      <Button
        outline
        color="danger"
        size="xs"
        className="btn btn-outline-danger rounded-pill"
        onClick={cancelButtonClickedHandler}
      >
        Cancel
      </Button>
    );
    const submitButton = (
      <Button
        outline
        color="primary"
        className="btn btn-outline-primary rounded-pill"
        onClick={postCommentHandler}
      >
        Comment
      </Button>
    );

    const isUploadable = isUploadableImage || isUploadableFile;

    return (
      <>
        <div className="comment-write">
          <CustomNavTab activeTab={activeTab} navTabMapping={navTabMapping} onNavSelected={handleSelect} hideBorderBottom />
          <TabContent activeTab={activeTab}>
            <TabPane tabId="comment_editor">
              <Editor
                ref={editorRef}
                value={comment}
                isUploadable={isUploadable}
                isUploadableFile={isUploadableFile}
                onChange={setComment}
                onUpload={uploadHandler}
                onCtrlEnter={ctrlEnterHandler}
                isComment
              />
              {/*
                Note: <OptionsSelector /> is not optimized for ComentEditor in terms of responsive design.
                See a review comment in https://github.com/weseek/growi/pull/3473
              */}
            </TabPane>
            <TabPane tabId="comment_preview">
              <div className="comment-form-preview">
                {commentPreview}
              </div>
            </TabPane>
          </TabContent>
        </div>

        <div className="comment-submit">
          <div className="d-flex">
            <span className="flex-grow-1" />
            <span className="d-none d-sm-inline">{ errorMessage && errorMessage }</span>

            { isSlackConfigured && isSlackEnabled != null
              && (
                <div className="form-inline align-self-center mr-md-2">
                  <SlackNotification
                    isSlackEnabled={isSlackEnabled}
                    slackChannels={slackChannels}
                    onEnabledFlagChange={isSlackEnabledToggleHandler}
                    onChannelChange={slackChannelsChangedHandler}
                    id="idForComment"
                  />
                </div>
              )
            }
            <div className="d-none d-sm-block">
              <span className="mr-2">{cancelButton}</span><span>{submitButton}</span>
            </div>
          </div>
          <div className="d-block d-sm-none mt-2">
            <div className="d-flex justify-content-end">
              { error && errorMessage }
              <span className="mr-2">{cancelButton}</span><span>{submitButton}</span>
            </div>
          </div>
        </div>
      </>
    );
  };

  return (
    <div className={`${styles['comment-editor-styles']} form page-comment-form`}>
      <div className="comment-form">
        <div className="comment-form-user">
          <UserPicture user={currentUser} noLink noTooltip />
        </div>
        <div className="comment-form-main">
          { isReadyToUse
            ? renderReady()
            : renderBeforeReady()
          }
        </div>
      </div>
    </div>
  );

};
