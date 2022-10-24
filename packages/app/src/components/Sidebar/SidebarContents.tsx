import React from 'react';

import { SidebarContentsType } from '~/interfaces/ui';
import { useCurrentSidebarContents } from '~/stores/ui';

import CustomSidebar from './CustomSidebar';
import PageTree from './PageTree';
import RecentChanges from './RecentChanges';
import Tag from './Tag';

export const SidebarContents = (): JSX.Element => {
  const { data: currentSidebarContents } = useCurrentSidebarContents();

  let Contents;
  switch (currentSidebarContents) {
    case SidebarContentsType.RECENT:
      Contents = RecentChanges;
      break;
    case SidebarContentsType.CUSTOM:
      Contents = CustomSidebar;
      break;
    case SidebarContentsType.TAG:
      Contents = Tag;
      break;
    default:
      Contents = PageTree;
  }

  return (
    <Contents />
  );

};
