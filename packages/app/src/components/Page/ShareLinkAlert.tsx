import React, { FC } from 'react';

import { useTranslation } from 'next-i18next';

const generateRatio = (expiredAt: Date, createdAt: Date): number => {
  const wholeTime = new Date(expiredAt).getTime() - new Date(createdAt).getTime();
  const remainingTime = new Date(expiredAt).getTime() - new Date().getTime();
  return remainingTime / wholeTime;
};

const getAlertColor = (ratio: number): string => {
  let color: string;

  if (ratio >= 0.75) {
    color = 'success';
  }
  else if (ratio < 0.75 && ratio >= 0.5) {
    color = 'info';
  }
  else if (ratio < 0.5 && ratio >= 0.25) {
    color = 'warning';
  }
  else {
    color = 'danger';
  }
  return color;
};

type Props = {
  createdAt: Date,
  expiredAt?: Date,
}

const ShareLinkAlert: FC<Props> = (props: Props) => {
  const { t } = useTranslation();
  const { expiredAt, createdAt } = props;

  const ratio = expiredAt != null ? generateRatio(expiredAt, createdAt) : 1;
  const alertColor = getAlertColor(ratio);

  return (
    <p className={`alert alert-${alertColor} my-3 px-4 d-edit-none`}>
      <i className="icon-fw icon-link"></i>
      {(expiredAt === null ? <span>{t('page_page.notice.no_deadline')}</span>
      // eslint-disable-next-line react/no-danger
        : <span dangerouslySetInnerHTML={{ __html: t('page_page.notice.expiration', { expiredAt }) }} />
      )}
    </p>
  );
};

export default ShareLinkAlert;
