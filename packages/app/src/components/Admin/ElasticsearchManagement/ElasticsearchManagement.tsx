import React, { useEffect, useState, useCallback } from 'react';

import { useTranslation } from 'next-i18next';


import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { apiv3Get, apiv3Post, apiv3Put } from '~/client/util/apiv3-client';
import { SocketEventName } from '~/interfaces/websocket';
import { useIsSearchServiceReachable } from '~/stores/context';
import { useAdminSocket } from '~/stores/socket-io';

import NormalizeIndicesControls from './NormalizeIndicesControls';
import RebuildIndexControls from './RebuildIndexControls';
import ReconnectControls from './ReconnectControls';
import StatusTable from './StatusTable';

const ElasticsearchManagement = () => {
  const { t } = useTranslation('admin');
  const { data: isSearchServiceReachable } = useIsSearchServiceReachable();
  const { data: socket } = useAdminSocket();

  const [isInitialized, setIsInitialized] = useState(false);

  const [isConnected, setIsConnected] = useState(false);
  const [isConfigured, setIsConfigured] = useState(false);
  const [isReconnectingProcessing, setIsReconnectingProcessing] = useState(false);
  const [isRebuildingProcessing, setIsRebuildingProcessing] = useState(false);
  const [isRebuildingCompleted, setIsRebuildingCompleted] = useState(false);

  const [isNormalized, setIsNormalized] = useState(false);
  const [indicesData, setIndicesData] = useState(null);
  const [aliasesData, setAliasesData] = useState(null);


  const retrieveIndicesStatus = useCallback(async() => {
    try {
      const { data } = await apiv3Get('/search/indices');
      const { info } = data;

      setIsConnected(true);
      setIsConfigured(true);

      setIndicesData(info.indices);
      setAliasesData(info.aliases);
      setIsNormalized(info.isNormalized);
    }
    catch (errors: unknown) {
      setIsConnected(false);

      // evaluate whether configured or not
      if (Array.isArray(errors)) {
        for (const error of errors) {
          if (error.code === 'search-service-unconfigured') {
            setIsConfigured(false);
          }
        }
      }

      toastError(errors);
    }
    finally {
      setIsInitialized(true);
    }
  }, []);

  useEffect(() => {
    const fetchIndicesStatusData = async() => {
      await retrieveIndicesStatus();
    };
    fetchIndicesStatusData();
  }, [retrieveIndicesStatus]);


  useEffect(() => {
    if (socket == null) {
      return;
    }
    socket.on(SocketEventName.AddPageProgress, (data) => {
      setIsRebuildingProcessing(true);
    });

    socket.on(SocketEventName.FinishAddPage, async(data) => {
      await retrieveIndicesStatus();
      setIsRebuildingProcessing(false);
      setIsRebuildingCompleted(true);
    });

    socket.on(SocketEventName.RebuildingFailed, (data) => {
      toastError(new Error(data.error), 'Rebuilding Index has failed.');
    });

    return () => {
      socket.off(SocketEventName.AddPageProgress);
      socket.off(SocketEventName.FinishAddPage);
      socket.off(SocketEventName.RebuildingFailed);
    };
  }, [socket]);


  const reconnect = async() => {
    setIsReconnectingProcessing(true);

    try {
      await apiv3Post('/search/connection');
    }
    catch (e) {
      toastError(e);
      return;
    }

    // reload
    window.location.reload();
  };

  const normalizeIndices = async() => {

    try {
      await apiv3Put('/search/indices', { operation: 'normalize' });
    }
    catch (e) {
      toastError(e);
    }

    await retrieveIndicesStatus();

    toastSuccess('Normalizing has succeeded');
  };

  const rebuildIndices = async() => {
    setIsRebuildingProcessing(true);

    try {
      await apiv3Put('/search/indices', { operation: 'rebuild' });
      toastSuccess('Rebuilding is requested');
    }
    catch (e) {
      toastError(e);
    }

    await retrieveIndicesStatus();
  };

  const isErrorOccuredOnSearchService = !isSearchServiceReachable;

  const isReconnectBtnEnabled = !isReconnectingProcessing && (!isInitialized || !isConnected || isErrorOccuredOnSearchService);

  return (
    <>
      <div data-testid="admin-full-text-search" className="row">
        <div className="col-md-12">
          <StatusTable
            isInitialized={isInitialized}
            isErrorOccuredOnSearchService={isErrorOccuredOnSearchService}
            isConnected={isConnected}
            isConfigured={isConfigured}
            isNormalized={isNormalized}
            indicesData={indicesData}
            aliasesData={aliasesData}
          />
        </div>
      </div>

      <hr />

      {/* Controls */}
      <div className="row">
        <label className="col-md-3 col-form-label text-left text-md-right">{ t('full_text_search_management.reconnect') }</label>
        <div className="col-md-6">
          <ReconnectControls
            isEnabled={isReconnectBtnEnabled}
            isProcessing={isReconnectingProcessing}
            onReconnectingRequested={reconnect}
          />
        </div>
      </div>

      <hr />

      <div className="row">
        <label className="col-md-3 col-form-label text-left text-md-right">{ t('full_text_search_management.normalize') }</label>
        <div className="col-md-6">
          <NormalizeIndicesControls
            isRebuildingProcessing={isRebuildingProcessing}
            isNormalized={isNormalized}
            onNormalizingRequested={normalizeIndices}
          />
        </div>
      </div>

      <hr />

      <div className="row">
        <label className="col-md-3 col-form-label text-left text-md-right">{ t('full_text_search_management.rebuild') }</label>
        <div className="col-md-6">
          <RebuildIndexControls
            isRebuildingProcessing={isRebuildingProcessing}
            isRebuildingCompleted={isRebuildingCompleted}
            isNormalized={isNormalized}
            onRebuildingRequested={rebuildIndices}
          />
        </div>
      </div>

    </>
  );

};


ElasticsearchManagement.propTypes = {

};

export default ElasticsearchManagement;
