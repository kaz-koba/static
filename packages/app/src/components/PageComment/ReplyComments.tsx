
import React, { useState } from 'react';

import { Collapse } from 'reactstrap';

import { RendererOptions } from '~/services/renderer/renderer';

import { ICommentHasId, ICommentHasIdList } from '../../interfaces/comment';
import { IUser } from '../../interfaces/user';
import { useIsAllReplyShown } from '../../stores/context';

import { Comment } from './Comment';

import styles from './ReplyComments.module.scss';


type ReplycommentsProps = {
  rendererOptions: RendererOptions,
  isReadOnly: boolean,
  revisionId: string,
  revisionCreatedAt: Date,
  currentUser: IUser,
  replyList: ICommentHasIdList,
  highlightKeywords?: string[],
  deleteBtnClicked: (comment: ICommentHasId) => void,
  onComment: () => void,
}

export const ReplyComments = (props: ReplycommentsProps): JSX.Element => {

  const {
    rendererOptions, isReadOnly, revisionId, revisionCreatedAt, currentUser, replyList, highlightKeywords,
    deleteBtnClicked, onComment,
  } = props;

  const { data: isAllReplyShown } = useIsAllReplyShown();

  const [isOlderRepliesShown, setIsOlderRepliesShown] = useState(false);

  const renderReply = (reply: ICommentHasId) => {
    return (
      <div key={reply._id} className={`${styles['page-comment-reply']} ml-4 ml-sm-5 mr-3`}>
        <Comment
          rendererOptions={rendererOptions}
          comment={reply}
          revisionId={revisionId}
          revisionCreatedAt={revisionCreatedAt}
          currentUser={currentUser}
          isReadOnly={isReadOnly}
          highlightKeywords={highlightKeywords}
          deleteBtnClicked={deleteBtnClicked}
          onComment={onComment}
        />
      </div>
    );
  };

  if (isAllReplyShown) {
    return (
      <>
        {replyList.map((reply) => {
          return renderReply(reply);
        })}
      </>
    );
  }

  const areThereHiddenReplies = (replyList.length > 2);
  const toggleButtonIconName = isOlderRepliesShown ? 'icon-arrow-up' : 'icon-options-vertical';
  const toggleButtonIcon = <i className={`icon-fw ${toggleButtonIconName}`}></i>;
  const toggleButtonLabel = isOlderRepliesShown ? '' : 'more';
  const shownReplies = replyList.slice(replyList.length - 2, replyList.length);
  const hiddenReplies = replyList.slice(0, replyList.length - 2);

  const hiddenElements = hiddenReplies.map((reply) => {
    return renderReply(reply);
  });

  const shownElements = shownReplies.map((reply) => {
    return renderReply(reply);
  });

  return (
    <>
      {areThereHiddenReplies && (
        <div className={`${styles['page-comments-hidden-replies']}`}>
          <Collapse isOpen={isOlderRepliesShown}>
            <div>{hiddenElements}</div>
          </Collapse>
          <div className="text-center">
            <button
              type="button"
              className="btn btn-link"
              onClick={() => setIsOlderRepliesShown(!isOlderRepliesShown)}
            >
              {toggleButtonIcon} {toggleButtonLabel}
            </button>
          </div>
        </div>
      )}
      {shownElements}
    </>
  );
};
