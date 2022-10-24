import React from 'react';

import { useGrowiVersion } from '~/stores/context';
import { useShortcutsModal } from '~/stores/modal';

import styles from './SystemVersion.module.scss';


type Props = {
  showShortcutsButton?: boolean,
}

const SystemVersion = (props: Props): JSX.Element => {
  const { showShortcutsButton } = props;

  const { open: openShortcutsModal } = useShortcutsModal();

  const { data: growiVersion } = useGrowiVersion();
  // add classes to cmd-key by OS
  const platform = window.navigator.platform.toLowerCase();
  const isMac = (platform.indexOf('mac') > -1);
  const os = isMac ? 'mac' : 'win';

  return (
    <>
      <div className={`${styles['system-version']} d-none d-md-flex d-edit-none d-print-none align-items-center`}>
        <span>
          <a href="https://growi.org">GROWI</a> {growiVersion}
        </span>
        { showShortcutsButton && (
          <button type="button" className="btn btn-link ml-2 p-0" onClick={() => openShortcutsModal()}>
            <i className="fa fa-keyboard-o"></i>&nbsp;<span className={`cmd-key ${os}`}></span>-/
          </button>
        ) }
      </div>

    </>
  );
};

export default SystemVersion;
