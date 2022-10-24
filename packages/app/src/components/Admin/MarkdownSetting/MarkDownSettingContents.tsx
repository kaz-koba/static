import React, { useEffect } from 'react';

import { useTranslation } from 'next-i18next';
import { Card, CardBody } from 'reactstrap';

import AdminMarkDownContainer from '~/client/services/AdminMarkDownContainer';
import { toastError } from '~/client/util/apiNotification';
import { toArrayIfNot } from '~/utils/array-utils';
import loggerFactory from '~/utils/logger';

import { withUnstatedContainers } from '../../UnstatedUtils';

import IndentForm from './IndentForm';
import LineBreakForm from './LineBreakForm';
import PresentationForm from './PresentationForm';
import XssForm from './XssForm';

const logger = loggerFactory('growi:MarkDown');

type Props ={
  adminMarkDownContainer: AdminMarkDownContainer
}

const MarkDownSettingContents = React.memo((props: Props): JSX.Element => {
  const { t } = useTranslation('admin');
  const { adminMarkDownContainer } = props;

  useEffect(() => {
    const fetchMarkdownData = async() => {
      await adminMarkDownContainer.retrieveMarkdownData();
    };

    try {
      fetchMarkdownData();
    }
    catch (err) {
      const errs = toArrayIfNot(err);
      toastError(errs);
      logger.error(errs);
    }
  }, [adminMarkDownContainer]);

  return (
    <div data-testid="admin-markdown">
      {/* Line Break Setting */}
      <h2 className="admin-setting-header">{t('markdown_settings.lineBreak_header')}</h2>
      <Card className="card well my-3">
        <CardBody className="px-0 py-2">{ t('markdown_settings.lineBreak_desc') }</CardBody>
      </Card>
      <LineBreakForm />

      {/* Indent Setting */}
      <h2 className="admin-setting-header">{t('markdown_settings.indent_header')}</h2>
      <Card className="card well my-3">
        <CardBody className="px-0 py-2">{t('markdown_settings.indent_desc') }</CardBody>
      </Card>
      <IndentForm />

      {/* Presentation Setting */}
      <h2 className="admin-setting-header">{ t('markdown_settings.presentation_header') }</h2>
      <Card className="card well my-3">
        <CardBody className="px-0 py-2">{ t('markdown_settings.presentation_desc') }</CardBody>
      </Card>
      <PresentationForm />

      {/* XSS Setting */}
      <h2 className="admin-setting-header">{ t('markdown_settings.xss_header') }</h2>
      <Card className="card well my-3">
        <CardBody className="px-0 py-2">{ t('markdown_settings.xss_desc') }</CardBody>
      </Card>
      <XssForm />
    </div>
  );
});
MarkDownSettingContents.displayName = 'MarkDownSettingContents';


const MarkdownSettingWithUnstatedContainer = withUnstatedContainers(MarkDownSettingContents, [AdminMarkDownContainer]);


export default MarkdownSettingWithUnstatedContainer;
