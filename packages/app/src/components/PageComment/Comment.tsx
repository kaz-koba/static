import React, { useEffect, useMemo, useState } from 'react';

import { IUser } from '@growi/core';
import { UserPicture } from '@growi/ui';
import { format } from 'date-fns';
import { useTranslation } from 'next-i18next';
import dynamic from 'next/dynamic';
import { UncontrolledTooltip } from 'reactstrap';

import { RendererOptions } from '~/services/renderer/renderer';

import { ICommentHasId } from '../../interfaces/comment';
import FormattedDistanceDate from '../FormattedDistanceDate';
import HistoryIcon from '../Icons/HistoryIcon';
import RevisionRenderer from '../Page/RevisionRenderer';
import Username from '../User/Username';

import { CommentControl } from './CommentControl';
import { CommentEditorProps } from './CommentEditor';

import styles from './Comment.module.scss';

const CommentEditor = dynamic<CommentEditorProps>(() => import('./CommentEditor').then(mod => mod.CommentEditor), { ssr: false });

type CommentProps = {
  comment: ICommentHasId,
  rendererOptions: RendererOptions,
  revisionId: string,
  revisionCreatedAt: Date,
  currentUser: IUser,
  isReadOnly: boolean,
  highlightKeywords?: string[],
  deleteBtnClicked: (comment: ICommentHasId) => void,
  onComment: () => void,
}

export const Comment = (props: CommentProps): JSX.Element => {

  const {
    comment, rendererOptions, revisionId, revisionCreatedAt, currentUser, isReadOnly,
    deleteBtnClicked, onComment,
  } = props;

  const { t } = useTranslation();

  const [markdown, setMarkdown] = useState('');
  const [isReEdit, setIsReEdit] = useState(false);

  const commentId = comment._id;
  const creator = comment.creator;
  const isMarkdown = comment.isMarkdown;
  const createdAt = new Date(comment.createdAt);
  const updatedAt = new Date(comment.updatedAt);
  const isEdited = createdAt < updatedAt;

  useEffect(() => {
    if (revisionId == null) {
      return;
    }

    setMarkdown(comment.comment);

    const isCurrentRevision = () => {
      return comment.revision === revisionId;
    };
    isCurrentRevision();
  }, [comment, revisionId]);

  const isCurrentUserEqualsToAuthor = () => {
    const { creator }: any = comment;
    if (creator == null || currentUser == null) {
      return false;
    }
    return creator.username === currentUser.username;
  };

  const getRootClassName = (comment: ICommentHasId) => {
    let className = 'page-comment flex-column';

    // Conditional for called from SearchResultContext
    if (revisionId != null && revisionCreatedAt != null) {
      if (comment.revision === revisionId) {
        className += ' page-comment-current';
      }
      else if (comment.createdAt.getTime() > revisionCreatedAt.getTime()) {
        className += ' page-comment-newer';
      }
      else {
        className += ' page-comment-older';
      }
    }

    if (isCurrentUserEqualsToAuthor()) {
      className += ' page-comment-me';
    }

    return className;
  };

  const deleteBtnClickedHandler = () => {
    deleteBtnClicked(comment);
  };

  const renderText = (comment: string) => {
    return <span style={{ whiteSpace: 'pre-wrap' }}>{comment}</span>;
  };

  const commentBody = useMemo(() => {
    if (rendererOptions == null) {
      return <></>;
    }

    return isMarkdown
      ? (
        <RevisionRenderer
          rendererOptions={rendererOptions}
          markdown={markdown}
          additionalClassName="comment"
        />
      )
      : renderText(comment.comment);
  }, [comment, isMarkdown, markdown, rendererOptions]);

  const rootClassName = getRootClassName(comment);
  const revHref = `?revision=${comment.revision}`;
  const editedDateId = `editedDate-${comment._id}`;
  const editedDateFormatted = isEdited ? format(updatedAt, 'yyyy/MM/dd HH:mm') : null;

  return (
    <div className={`${styles['comment-styles']}`}>
      { (isReEdit && !isReadOnly) ? (
        <CommentEditor
          pageId={comment._id}
          replyTo={undefined}
          currentCommentId={commentId}
          commentBody={comment.comment}
          onCancelButtonClicked={() => setIsReEdit(false)}
          onCommentButtonClicked={() => {
            setIsReEdit(false);
            if (onComment != null) onComment();
          }}
        />
      ) : (
        <div id={commentId} className={rootClassName}>
          <div className="page-comment-writer">
            <UserPicture user={creator} />
          </div>
          <div className="page-comment-main">
            <div className="page-comment-creator">
              <Username user={creator} />
            </div>
            <div className="page-comment-body">{commentBody}</div>
            <div className="page-comment-meta">
              <a href={`#${commentId}`}>
                <FormattedDistanceDate id={commentId} date={comment.createdAt} />
              </a>
              { isEdited && (
                <>
                  <span id={editedDateId}>&nbsp;(edited)</span>
                  <UncontrolledTooltip placement="bottom" fade={false} target={editedDateId}>{editedDateFormatted}</UncontrolledTooltip>
                </>
              ) }
              <span className="ml-2">
                <a id={`page-comment-revision-${commentId}`} className="page-comment-revision" href={revHref}>
                  <HistoryIcon />
                </a>
                <UncontrolledTooltip placement="bottom" fade={false} target={`page-comment-revision-${commentId}`}>
                  {t('page_comment.display_the_page_when_posting_this_comment')}
                </UncontrolledTooltip>
              </span>
            </div>
            { (isCurrentUserEqualsToAuthor() && !isReadOnly) && (
              <CommentControl
                onClickDeleteBtn={deleteBtnClickedHandler}
                onClickEditBtn={() => setIsReEdit(true)}
              />
            ) }
          </div>
        </div>
      ) }
    </div>
  );
};
