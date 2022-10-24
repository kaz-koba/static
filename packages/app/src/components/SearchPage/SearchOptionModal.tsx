import React, { FC } from 'react';
import { useTranslation } from 'next-i18next';

import {
  Modal, ModalHeader, ModalBody,
} from 'reactstrap';


type Props = {
  isOpen: boolean,
  includeUserPages: boolean,
  includeTrashPages: boolean,
  onClose?: () => void,
  onIncludeUserPagesSwitched?: (isChecked: boolean) => void,
  onIncludeTrashPagesSwitched?: (isChecked: boolean) => void,
}

const SearchOptionModal: FC<Props> = (props: Props) => {

  const { t } = useTranslation('');

  const {
    isOpen, includeUserPages, includeTrashPages,
    onClose,
    onIncludeUserPagesSwitched,
    onIncludeTrashPagesSwitched,
  } = props;

  const onCloseModal = () => {
    if (onClose != null) {
      onClose();
    }
  };

  const includeUserPagesChangeHandler = (isChecked: boolean) => {
    if (onIncludeUserPagesSwitched != null) {
      onIncludeUserPagesSwitched(isChecked);
    }
  };

  const includeTrashPagesChangeHandler = (isChecked: boolean) => {
    if (onIncludeTrashPagesSwitched != null) {
      onIncludeTrashPagesSwitched(isChecked);
    }
  };

  return (
    <Modal size="lg" isOpen={isOpen} toggle={onCloseModal} autoFocus={false}>
      <ModalHeader tag="h4" toggle={onCloseModal} className="bg-primary text-light">
        Search Option
      </ModalHeader>
      <ModalBody>
        <div className="d-flex p-2">
          <div className="border border-gray mr-3">
            <label className="px-3 py-2 mb-0 d-flex align-items-center">
              <input
                className="mr-2"
                type="checkbox"
                onChange={e => includeUserPagesChangeHandler(e.target.checked)}
                checked={includeUserPages}
              />
              {t('Include Subordinated Target Page', { target: '/user' })}
            </label>
          </div>
          <div className="border border-gray">
            <label className="px-3 py-2 mb-0 d-flex align-items-center">
              <input
                className="mr-2"
                type="checkbox"
                onChange={e => includeTrashPagesChangeHandler(e.target.checked)}
                checked={includeTrashPages}
              />
              {t('Include Subordinated Target Page', { target: '/trash' })}
            </label>
          </div>
        </div>
      </ModalBody>
    </Modal>
  );
};

export default SearchOptionModal;
