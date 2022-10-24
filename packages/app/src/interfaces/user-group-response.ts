import { HasObjectId, Ref } from '@growi/core';

import { IPageHasId } from './page';
import {
  IUser, IUserGroup, IUserGroupHasId, IUserGroupRelationHasId,
} from './user';

export type UserGroupResult = {
  userGroup: IUserGroupHasId,
}

export type UserGroupListResult = {
  userGroups: IUserGroupHasId[],
};

export type ChildUserGroupListResult = {
  childUserGroups: IUserGroupHasId[],
  grandChildUserGroups: IUserGroupHasId[],
};

export type UserGroupRelationListResult = {
  userGroupRelations: IUserGroupRelationHasId[],
};

export type IUserGroupRelationHasIdPopulatedUser = {
  relatedGroup: Ref<IUserGroup>,
  relatedUser: IUser & HasObjectId,
  createdAt: Date,
} & HasObjectId;

export type UserGroupRelationsResult = {
  userGroupRelations: IUserGroupRelationHasIdPopulatedUser[],
};

export type UserGroupPagesResult = {
  pages: IPageHasId[],
}

export type SelectableParentUserGroupsResult = {
  selectableParentGroups: IUserGroupHasId[],
}

export type SelectableUserChildGroupsResult = {
  selectableChildGroups: IUserGroupHasId[],
}

export type AncestorUserGroupsResult = {
  ancestorUserGroups: IUserGroupHasId[],
}
