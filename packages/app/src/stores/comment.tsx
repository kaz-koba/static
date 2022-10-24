import { Nullable } from '@growi/core';
import useSWR, { SWRResponse } from 'swr';

import { apiGet, apiPost } from '~/client/util/apiv1-client';

import { ICommentHasIdList, ICommentPostArgs } from '../interfaces/comment';

type IResponseComment = {
  comments: ICommentHasIdList,
  ok: boolean,
}

type CommentOperation = {
  update(comment: string, revisionId: string, commentId: string): Promise<void>,
  post(args: ICommentPostArgs): Promise<void>
}

export const useSWRxPageComment = (pageId: Nullable<string>): SWRResponse<ICommentHasIdList, Error> & CommentOperation => {
  const shouldFetch: boolean = pageId != null;

  const swrResponse = useSWR(
    shouldFetch ? ['/comments.get', pageId] : null,
    (endpoint, pageId) => apiGet(endpoint, { page_id: pageId }).then((response:IResponseComment) => response.comments),
  );

  const update = async(comment: string, revisionId: string, commentId: string) => {
    const { mutate } = swrResponse;
    await apiPost('/comments.update', {
      commentForm: {
        comment,
        revision_id: revisionId,
        comment_id: commentId,
      },
    });
    mutate();
  };

  const post = async(args: ICommentPostArgs) => {
    const { mutate } = swrResponse;
    const { commentForm, slackNotificationForm } = args;
    const { comment, revisionId, replyTo } = commentForm;
    const { isSlackEnabled, slackChannels } = slackNotificationForm;

    await apiPost('/comments.add', {
      commentForm: {
        comment,
        page_id: pageId,
        revision_id: revisionId,
        replyTo,
      },
      slackNotificationForm: {
        isSlackEnabled,
        slackChannels,
      },
    });
    mutate();
  };

  return {
    ...swrResponse,
    update,
    post,
  };
};
