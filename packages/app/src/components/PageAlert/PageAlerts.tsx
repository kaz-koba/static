import React from 'react';

import dynamic from 'next/dynamic';

import { useIsNotFound } from '~/stores/context';

import { FixPageGrantAlert } from './FixPageGrantAlert';
import { OldRevisionAlert } from './OldRevisionAlert';
import { PageGrantAlert } from './PageGrantAlert';
import { PageRedirectedAlert } from './PageRedirectedAlert';
import { PageStaleAlert } from './PageStaleAlert';

// dynamic import because TrashPageAlert uses localStorageMiddleware
const TrashPageAlert = dynamic(() => import('./TrashPageAlert').then(mod => mod.TrashPageAlert), { ssr: false });

export const PageAlerts = (): JSX.Element => {

  const { data: isNotFound } = useIsNotFound();

  return (
    <div className="row d-edit-none">
      <div className="col-sm-12">
        {/* alerts */}
        { !isNotFound && <FixPageGrantAlert /> }
        <PageGrantAlert />
        <TrashPageAlert />
        <PageStaleAlert />
        <OldRevisionAlert />
        <PageRedirectedAlert />
      </div>
    </div>
  );
};
