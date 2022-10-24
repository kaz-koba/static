import React, {
  Fragment, useState, useCallback, useRef, ForwardRefRenderFunction, forwardRef, useImperativeHandle,
} from 'react';

import { AsyncTypeahead, Menu, MenuItem } from 'react-bootstrap-typeahead';
import { useTranslation } from 'react-i18next';

import { IClearable } from '~/client/interfaces/clearable';
import { useSWRxUsernames } from '~/stores/user';


const Categories = {
  activeUser: 'Active User',
  inactiveUser: 'Inactive User',
  activitySnapshotUser: 'Activity Snapshot User',
} as const;

type CategoryType = typeof Categories[keyof typeof Categories]

type UserDataType = {
  username: string
  category: CategoryType
}

type Props = {
  onChange: (text: string[]) => void
}

const SearchUsernameTypeaheadSubstance: ForwardRefRenderFunction<IClearable, Props> = ((props: Props, ref) => {
  const { onChange } = props;
  const { t } = useTranslation();

  const typeaheadRef = useRef<IClearable>(null);

  /*
   * State
   */
  const [searchKeyword, setSearchKeyword] = useState<string>('');

  /*
   * Fetch
   */
  const requestOptions = { isIncludeActiveUser: true, isIncludeInactiveUser: true, isIncludeActivitySnapshotUser: true };
  const { data: usernameData, error } = useSWRxUsernames(searchKeyword, 0, 5, requestOptions);
  const activeUsernames = usernameData?.activeUser?.usernames != null ? usernameData.activeUser.usernames : [];
  const inactiveUsernames = usernameData?.inactiveUser?.usernames != null ? usernameData.inactiveUser.usernames : [];
  const activitySnapshotUsernames = usernameData?.activitySnapshotUser?.usernames != null ? usernameData.activitySnapshotUser.usernames : [];
  const isLoading = usernameData === undefined && error == null;

  const allUser: UserDataType[] = [];
  const pushToAllUser = (usernames: string[], category: CategoryType) => {
    usernames.forEach(username => allUser.push({ username, category }));
  };
  pushToAllUser(activeUsernames, Categories.activeUser);
  pushToAllUser(inactiveUsernames, Categories.inactiveUser);
  pushToAllUser(activitySnapshotUsernames, Categories.activitySnapshotUser);

  /*
   * Functions
   */
  const changeHandler = useCallback((userData: UserDataType[]) => {
    if (onChange != null) {
      const usernames = userData.map(user => user.username);
      onChange(usernames);
    }
  }, [onChange]);

  const searchHandler = useCallback((text: string) => {
    setSearchKeyword(text);
  }, []);

  const renderMenu = useCallback((allUser: UserDataType[], menuProps) => {
    if (allUser == null || allUser.length === 0) {
      return <></>;
    }

    let index = 0;
    const items = Object.values(Categories).map((category) => {
      const userData = allUser.filter(user => user.category === category);
      return (
        <Fragment key={category}>
          {index !== 0 && <Menu.Divider />}
          <Menu.Header>{category}</Menu.Header>
          {userData.map((user) => {
            const item = (
              <MenuItem key={index} option={user} position={index}>
                {user.username}
              </MenuItem>
            );
            index++;
            return item;
          })}
        </Fragment>
      );
    });

    return (
      <Menu {...menuProps}>{items}</Menu>
    );
  }, []);

  useImperativeHandle(ref, () => ({
    clear() {
      const instance = typeaheadRef?.current;
      if (instance != null) {
        instance.clear();
      }
    },
  }));

  return (
    <div className="input-group mr-2">
      <div className="input-group-prepend">
        <span className="input-group-text">
          <i className="icon-people" />
        </span>
      </div>
      <AsyncTypeahead
        ref={typeaheadRef}
        id="search-username-typeahead-asynctypeahead"
        multiple
        delay={400}
        minLength={0}
        placeholder={t('admin:audit_log_management.username')}
        caseSensitive={false}
        isLoading={isLoading}
        options={allUser}
        onSearch={searchHandler}
        onChange={changeHandler}
        renderMenu={renderMenu}
        labelKey={(option: UserDataType) => `${option.username}`}
      />
    </div>
  );
});

export const SearchUsernameTypeahead = forwardRef(SearchUsernameTypeaheadSubstance);
