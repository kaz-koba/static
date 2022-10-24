import React from 'react';

import { IUserHasId } from '@growi/core';
import { UserPicture } from '@growi/ui';

import styles from './UserInfo.module.scss';


export type UserInfoProps = {
  author?: IUserHasId,
}

export const UserInfo = (props: UserInfoProps): JSX.Element => {

  const { author } = props;

  if (author == null || author.status === 4) {
    return <></>;
  }

  return (
    <div className={`${styles['grw-users-info']} grw-users-info d-flex align-items-center d-edit-none mb-5 pb-3 border-bottom`}>
      <UserPicture user={author} />
      <div className="users-meta">
        <h1 className="user-page-name">
          {author.name}
        </h1>
        <div className="user-page-meta mt-3 mb-0">
          <span className="user-page-username mr-4"><i className="icon-user mr-1"></i>{author.username}</span>
          <span className="user-page-email mr-2">
            <i className="icon-envelope mr-1"></i>
            { author.isEmailPublished
              ? author.email
              : '*****'
            }
          </span>
          { author.introduction && (
            <span className="user-page-introduction">{author.introduction}</span>
          ) }
        </div>
      </div>
    </div>
  );

};
