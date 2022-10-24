import React, { FC, useState, useCallback } from 'react';

import { isInteger } from 'core-js/fn/number';
import {
  format, parse, addDays, set,
} from 'date-fns';
import { useTranslation } from 'next-i18next';

import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { apiv3Post } from '~/client/util/apiv3-client';
import { useCurrentPageId } from '~/stores/context';


const ExpirationType = {
  UNLIMITED: 'unlimited',
  CUSTOM: 'custom',
  NUMBER_OF_DAYS: 'numberOfDays',
} as const;

type ExpirationType = typeof ExpirationType[keyof typeof ExpirationType];

type Props = {
  onCloseForm: () => void,
}

export const ShareLinkForm: FC<Props> = (props: Props) => {
  const { t } = useTranslation();
  const { onCloseForm } = props;

  const [expirationType, setExpirationType] = useState<ExpirationType>(ExpirationType.UNLIMITED);
  const [numberOfDays, setNumberOfDays] = useState<number>(7);
  const [description, setDescription] = useState<string>('');
  const [customExpirationDate, setCustomExpirationDate] = useState<Date>(new Date());
  const [customExpirationTime, setCustomExpirationTime] = useState<Date>(new Date());

  const { data: currentPageId } = useCurrentPageId();

  const handleChangeExpirationType = useCallback((expirationType: ExpirationType) => {
    setExpirationType(expirationType);
  }, []);

  const handleChangeNumberOfDays = useCallback((numberOfDays: number) => {
    setNumberOfDays(numberOfDays);
  }, []);

  const handleChangeDescription = useCallback((description: string) => {
    setDescription(description);
  }, []);

  const handleChangeCustomExpirationDate = useCallback((customExpirationDate: string) => {
    const parsedDate = parse(customExpirationDate, 'yyyy-MM-dd', new Date());
    setCustomExpirationDate(parsedDate);
  }, []);

  const handleChangeCustomExpirationTime = useCallback((customExpirationTime: string) => {
    const parsedTime = parse(customExpirationTime, 'HH:mm', new Date());
    setCustomExpirationTime(parsedTime);
  }, []);

  const generateExpired = useCallback(() => {
    if (expirationType === ExpirationType.UNLIMITED) {
      return null;
    }

    if (expirationType === ExpirationType.NUMBER_OF_DAYS) {
      if (!isInteger(Number(numberOfDays))) {
        throw new Error(t('share_links.Invalid_Number_of_Date'));
      }
      return addDays(new Date(), numberOfDays);
    }

    if (expirationType === ExpirationType.CUSTOM) {
      return set(customExpirationDate, { hours: customExpirationTime.getHours(), minutes: customExpirationTime.getMinutes() });
    }
  }, [t, customExpirationTime, customExpirationDate, expirationType, numberOfDays]);

  const closeForm = useCallback(() => {
    if (onCloseForm == null) {
      return;
    }
    onCloseForm();
  }, [onCloseForm]);

  const handleIssueShareLink = useCallback(async() => {
    let expiredAt;

    try {
      expiredAt = generateExpired();
    }
    catch (err) {
      return toastError(err);
    }

    try {
      await apiv3Post('/share-links/', { relatedPage: currentPageId, expiredAt, description });
      closeForm();
      toastSuccess(t('toaster.issue_share_link'));
    }
    catch (err) {
      toastError(err);
    }
  }, [t, currentPageId, description, closeForm, generateExpired]);

  return (
    <div className="share-link-form p-3">
      <h3 className="grw-modal-head pb-2"> { t('share_links.share_settings') }</h3>
      <div className=" p-3">

        {/* ExpirationTypeOptions */}
        <div className="form-group row">
          <label htmlFor="inputDesc" className="col-md-5 col-form-label">{t('share_links.expire')}</label>
          <div className="col-md-7">

            <div className="custom-control custom-radio form-group ">
              <input
                type="radio"
                className="custom-control-input"
                id="customRadio1"
                name="expirationType"
                value="customRadio1"
                checked={expirationType === ExpirationType.UNLIMITED}
                onChange={() => { handleChangeExpirationType(ExpirationType.UNLIMITED) }}
              />
              <label className="custom-control-label" htmlFor="customRadio1">{t('share_links.Unlimited')}</label>
            </div>

            <div className="custom-control custom-radio  form-group">
              <input
                type="radio"
                className="custom-control-input"
                id="customRadio2"
                value="customRadio2"
                checked={expirationType === ExpirationType.NUMBER_OF_DAYS}
                onChange={() => { handleChangeExpirationType(ExpirationType.NUMBER_OF_DAYS) }}
                name="expirationType"
              />
              <label className="custom-control-label" htmlFor="customRadio2">
                <div className="row align-items-center m-0">
                  <input
                    type="number"
                    min="1"
                    className="col-4"
                    name="expirationType"
                    value={numberOfDays}
                    onFocus={() => { handleChangeExpirationType(ExpirationType.NUMBER_OF_DAYS) }}
                    onChange={e => handleChangeNumberOfDays(Number(e.target.value))}
                  />
                  <span className="col-auto">{t('share_links.Days')}</span>
                </div>
              </label>
            </div>

            <div className="custom-control custom-radio form-group text-nowrap mb-0">
              <input
                type="radio"
                className="custom-control-input"
                id="customRadio3"
                name="expirationType"
                value="customRadio3"
                checked={expirationType === ExpirationType.CUSTOM}
                onChange={() => { handleChangeExpirationType(ExpirationType.CUSTOM) }}
              />
              <label className="custom-control-label" htmlFor="customRadio3">
                {t('share_links.Custom')}
              </label>
              <div className="d-inline-flex flex-wrap">
                <input
                  type="date"
                  className="ml-3 mb-2"
                  name="customExpirationDate"
                  value={format(customExpirationDate, 'yyyy-MM-dd')}
                  onFocus={() => { handleChangeExpirationType(ExpirationType.CUSTOM) }}
                  onChange={e => handleChangeCustomExpirationDate(e.target.value)}
                />
                <input
                  type="time"
                  className="ml-3 mb-2"
                  name="customExpiration"
                  value={format(customExpirationTime, 'HH:mm')}
                  onFocus={() => { handleChangeExpirationType(ExpirationType.CUSTOM) }}
                  onChange={e => handleChangeCustomExpirationTime(e.target.value)}
                />
              </div>
            </div>
          </div>
        </div>

        {/* DescriptionForm */}
        <div className="form-group row">
          <label htmlFor="inputDesc" className="col-md-5 col-form-label">{t('share_links.description')}</label>
          <div className="col-md-4">
            <input
              type="text"
              className="form-control"
              id="inputDesc"
              placeholder={t('share_links.enter_desc')}
              value={description}
              onChange={e => handleChangeDescription(e.target.value)}
            />
          </div>
        </div>
        <button type="button" className="btn btn-primary d-block mx-auto px-5" onClick={handleIssueShareLink}>
          {t('share_links.Issue')}
        </button>
      </div>
    </div>
  );
};
