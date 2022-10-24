// eslint-disable-next-line no-restricted-imports
import { AxiosResponse } from 'axios';

import { debounce } from 'throttle-debounce';

import { apiv3Put } from '~/client/util/apiv3-client';
import { IUserUISettings } from '~/interfaces/user-ui-settings';
import { useIsGuestUser } from '~/stores/context';

let settingsForBulk: Partial<IUserUISettings> = {};
const _putUserUISettingsInBulk = (): Promise<AxiosResponse<IUserUISettings>> => {
  const result = apiv3Put<IUserUISettings>('/user-ui-settings', { settings: settingsForBulk });

  // clear partial
  settingsForBulk = {};

  return result;
};

const _putUserUISettingsInBulkDebounced = debounce(1500, false, _putUserUISettingsInBulk);

type ScheduleToPutFunction = (settings: Partial<IUserUISettings>) => Promise<AxiosResponse<IUserUISettings>>;
const scheduleToPut: ScheduleToPutFunction = (settings: Partial<IUserUISettings>): Promise<AxiosResponse<IUserUISettings>> => {
  settingsForBulk = {
    ...settingsForBulk,
    ...settings,
  };

  return _putUserUISettingsInBulkDebounced();
};

type UserUISettingsUtil = {
  scheduleToPut: ScheduleToPutFunction | (() => void),
}
export const useUserUISettings = (): UserUISettingsUtil => {
  const { data: isGuestUser } = useIsGuestUser();

  return {
    scheduleToPut: isGuestUser
      ? () => {}
      : scheduleToPut,
  };
};
