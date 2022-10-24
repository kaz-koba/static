import React, {
  useEffect, useRef, useState, useMemo, useCallback,
} from 'react';

import { Nullable } from '@growi/core';
import { useTranslation } from 'next-i18next';
import { debounce } from 'throttle-debounce';

import { toastError, toastSuccess } from '~/client/util/apiNotification';
import { IPageHasId, IPageToDeleteWithMeta } from '~/interfaces/page';
import { AncestorsChildrenResult, RootPageResult, TargetAndAncestors } from '~/interfaces/page-listing-results';
import { OnDuplicatedFunction, OnDeletedFunction } from '~/interfaces/ui';
import { SocketEventName, UpdateDescCountData, UpdateDescCountRawData } from '~/interfaces/websocket';
import { useIsEnabledAttachTitleHeader } from '~/stores/context';
import {
  IPageForPageDuplicateModal, usePageDuplicateModal, usePageDeleteModal,
} from '~/stores/modal';
import {
  usePageTreeTermManager, useSWRxPageAncestorsChildren, useSWRxRootPage, useDescendantsPageListForCurrentPathTermManager,
} from '~/stores/page-listing';
import { useFullTextSearchTermManager } from '~/stores/search';
import { usePageTreeDescCountMap, useSidebarScrollerRef } from '~/stores/ui';
import { useGlobalSocket } from '~/stores/websocket';
import loggerFactory from '~/utils/logger';

import Item from './Item';
import { ItemNode } from './ItemNode';

import styles from './ItemsTree.module.scss';

const logger = loggerFactory('growi:cli:ItemsTree');

/*
 * Utility to generate initial node
 */
const generateInitialNodeBeforeResponse = (targetAndAncestors: Partial<IPageHasId>[]): ItemNode => {
  const nodes = targetAndAncestors.map((page): ItemNode => {
    return new ItemNode(page, []);
  });

  // update children for each node
  const rootNode = nodes.reduce((child, parent) => {
    parent.children = [child];
    return parent;
  });

  return rootNode;
};

const generateInitialNodeAfterResponse = (ancestorsChildren: Record<string, Partial<IPageHasId>[]>, rootNode: ItemNode): ItemNode => {
  const paths = Object.keys(ancestorsChildren);

  let currentNode = rootNode;
  paths.every((path) => {
    // stop rendering when non-migrated pages found
    if (currentNode == null) {
      return false;
    }

    const childPages = ancestorsChildren[path];
    currentNode.children = ItemNode.generateNodesFromPages(childPages);
    const nextNode = currentNode.children.filter((node) => {
      return paths.includes(node.page.path as string);
    })[0];
    currentNode = nextNode;
    return true;
  });

  return rootNode;
};

// user defined typeguard to assert the arg is not null
type RenderingCondition = {
  ancestorsChildrenResult: AncestorsChildrenResult | undefined,
  rootPageResult: RootPageResult | undefined,
}
type SecondStageRenderingCondition = {
  ancestorsChildrenResult: AncestorsChildrenResult,
  rootPageResult: RootPageResult,
}
const isSecondStageRenderingCondition = (condition: RenderingCondition|SecondStageRenderingCondition): condition is SecondStageRenderingCondition => {
  return condition.ancestorsChildrenResult != null && condition.rootPageResult != null;
};


type ItemsTreeProps = {
  isEnableActions: boolean
  targetPath: string
  targetPathOrId?: Nullable<string>
  targetAndAncestorsData?: TargetAndAncestors
}

/*
 * ItemsTree
 */
const ItemsTree = (props: ItemsTreeProps): JSX.Element => {
  const {
    targetPath, targetPathOrId, targetAndAncestorsData, isEnableActions,
  } = props;

  const { t } = useTranslation();

  const { data: ancestorsChildrenResult, error: error1 } = useSWRxPageAncestorsChildren(targetPath);
  const { data: rootPageResult, error: error2 } = useSWRxRootPage();
  const { data: isEnabledAttachTitleHeader } = useIsEnabledAttachTitleHeader();
  const { open: openDuplicateModal } = usePageDuplicateModal();
  const { open: openDeleteModal } = usePageDeleteModal();
  const { data: sidebarScrollerRef } = useSidebarScrollerRef();

  const { data: socket } = useGlobalSocket();
  const { data: ptDescCountMap, update: updatePtDescCountMap } = usePageTreeDescCountMap();

  // for mutation
  const { advance: advancePt } = usePageTreeTermManager();
  const { advance: advanceFts } = useFullTextSearchTermManager();
  const { advance: advanceDpl } = useDescendantsPageListForCurrentPathTermManager();

  const [isInitialScrollCompleted, setIsInitialScrollCompleted] = useState(false);

  const rootElemRef = useRef(null);

  const renderingCondition = useMemo(() => {
    return {
      ancestorsChildrenResult,
      rootPageResult,
    };
  }, [ancestorsChildrenResult, rootPageResult]);

  useEffect(() => {
    if (socket == null) {
      return;
    }

    socket.on(SocketEventName.UpdateDescCount, (data: UpdateDescCountRawData) => {
      // save to global state
      const newData: UpdateDescCountData = new Map(Object.entries(data));

      updatePtDescCountMap(newData);
    });

    return () => { socket.off(SocketEventName.UpdateDescCount) };

  }, [socket, ptDescCountMap, updatePtDescCountMap]);

  const onRenamed = () => {
    advancePt();
    advanceFts();
    advanceDpl();
  };

  const onClickDuplicateMenuItem = (pageToDuplicate: IPageForPageDuplicateModal) => {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const duplicatedHandler: OnDuplicatedFunction = (fromPath, toPath) => {
      toastSuccess(t('duplicated_pages', { fromPath }));

      advancePt();
      advanceFts();
      advanceDpl();
    };

    openDuplicateModal(pageToDuplicate, { onDuplicated: duplicatedHandler });
  };

  const onClickDeleteMenuItem = (pageToDelete: IPageToDeleteWithMeta) => {
    const onDeletedHandler: OnDeletedFunction = (pathOrPathsToDelete, isRecursively, isCompletely) => {
      if (typeof pathOrPathsToDelete !== 'string') {
        return;
      }

      const path = pathOrPathsToDelete;

      if (isCompletely) {
        toastSuccess(t('deleted_pages_completely', { path }));
      }
      else {
        toastSuccess(t('deleted_pages', { path }));
      }

      advancePt();
      advanceFts();
      advanceDpl();
    };

    openDeleteModal([pageToDelete], { onDeleted: onDeletedHandler });
  };

  // ***************************  Scroll on init ***************************
  const scrollOnInit = useCallback(() => {
    const scrollTargetElement = document.getElementById('grw-pagetree-current-page-item');

    if (sidebarScrollerRef?.current == null || scrollTargetElement == null) {
      return;
    }

    logger.debug('scrollOnInit has invoked');

    const scrollElement = sidebarScrollerRef.current.getScrollElement();

    // NOTE: could not use scrollIntoView
    //  https://stackoverflow.com/questions/11039885/scrollintoview-causing-the-whole-page-to-move

    // calculate the center point
    const scrollTop = scrollTargetElement.offsetTop - scrollElement.getBoundingClientRect().height / 2;
    scrollElement.scrollTo({ top: scrollTop });

    setIsInitialScrollCompleted(true);
  }, [sidebarScrollerRef]);

  const scrollOnInitDebounced = useMemo(() => debounce(500, scrollOnInit), [scrollOnInit]);

  useEffect(() => {
    if (!isSecondStageRenderingCondition(renderingCondition) || isInitialScrollCompleted) {
      return;
    }

    const rootElement = rootElemRef.current as HTMLElement | null;
    if (rootElement == null) {
      return;
    }

    const observerCallback = (mutationRecords: MutationRecord[]) => {
      mutationRecords.forEach(() => scrollOnInitDebounced());
    };

    const observer = new MutationObserver(observerCallback);
    observer.observe(rootElement, { childList: true, subtree: true });

    // first call for the situation that all rendering is complete at this point
    scrollOnInitDebounced();

    return () => {
      observer.disconnect();
    };
  }, [isInitialScrollCompleted, renderingCondition, scrollOnInitDebounced]);
  // *******************************  end  *******************************

  if (error1 != null || error2 != null) {
    // TODO: improve message
    toastError('Error occurred while fetching pages to render PageTree');
    return <></>;
  }

  let initialItemNode;
  /*
   * Render second stage
   */
  if (isSecondStageRenderingCondition(renderingCondition)) {
    initialItemNode = generateInitialNodeAfterResponse(
      renderingCondition.ancestorsChildrenResult.ancestorsChildren,
      new ItemNode(renderingCondition.rootPageResult.rootPage),
    );
  }
  /*
   * Before swr response comes back
   */
  else if (targetAndAncestorsData != null) {
    initialItemNode = generateInitialNodeBeforeResponse(targetAndAncestorsData.targetAndAncestors);
  }

  if (initialItemNode != null) {
    return (
      <ul className={`grw-pagetree ${styles['grw-pagetree']} list-group p-3`} ref={rootElemRef}>
        <Item
          key={initialItemNode.page.path}
          targetPathOrId={targetPathOrId}
          itemNode={initialItemNode}
          isOpen
          isEnabledAttachTitleHeader={isEnabledAttachTitleHeader}
          isEnableActions={isEnableActions}
          onRenamed={onRenamed}
          onClickDuplicateMenuItem={onClickDuplicateMenuItem}
          onClickDeleteMenuItem={onClickDeleteMenuItem}
        />
      </ul>
    );
  }

  return <></>;
};

export default ItemsTree;
