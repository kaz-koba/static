import { Middleware, SWRHook } from 'swr';

import { IUserHasId } from '~/interfaces/user';

import { apiv3Put } from '~/client/util/apiv3-client';

export const checkAndUpdateImageUrlCached: Middleware = (useSWRNext: SWRHook) => {
  return (key, fetcher, config) => {
    const swrNext = useSWRNext(key, fetcher, config);
    if (swrNext.data != null) {

      const userIds = Object(swrNext.data)
        .filter((user: IUserHasId) => user.imageUrlCached == null)
        .map((user: IUserHasId) => user._id);

      if (userIds.length > 0) {
        const distinctUserIds = Array.from(new Set(userIds));
        apiv3Put('/users/update.imageUrlCache', { userIds: distinctUserIds });
      }
    }
    return swrNext;
  };
};
