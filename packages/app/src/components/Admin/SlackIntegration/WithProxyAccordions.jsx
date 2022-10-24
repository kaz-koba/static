/* eslint-disable react/prop-types */
import React, { useState, useCallback } from 'react';

import { SlackbotType } from '@growi/slack';
import { useTranslation } from 'next-i18next';
import PropTypes from 'prop-types';
import { CopyToClipboard } from 'react-copy-to-clipboard';
import { Tooltip } from 'reactstrap';

import { toastSuccess, toastError } from '~/client/util/apiNotification';
import { apiv3Put, apiv3Post } from '~/client/util/apiv3-client';
import { useSiteUrl } from '~/stores/context';
import loggerFactory from '~/utils/logger';

import { withUnstatedContainers } from '../../UnstatedUtils';
import Accordion from '../Common/Accordion';

import ManageCommandsProcess from './ManageCommandsProcess';
import MessageBasedOnConnection from './MessageBasedOnConnection';
import { addLogs } from './slak-integration-util';

const logger = loggerFactory('growi:SlackIntegration:WithProxyAccordionsWrapper');

const BotCreateProcess = () => {
  const { t } = useTranslation();
  return (
    <div className="my-5 d-flex flex-column align-items-center">
      <button type="button" className="btn btn-primary text-nowrap" onClick={() => window.open('https://api.slack.com/apps', '_blank')}>
        {t('admin:slack_integration.accordion.create_bot')}
        <i className="fa fa-external-link ml-2" aria-hidden="true" />
      </button>
      <a
        href={t('admin:slack_integration.docs_url.custom_bot_with_proxy_setting')}
        target="_blank"
        rel="noopener noreferrer"
      >
        <p className="text-center mt-1">
          <small>
            {t('admin:slack_integration.accordion.how_to_create_a_bot')}
            <i className="fa fa-external-link ml-2" aria-hidden="true" />
          </small>
        </p>
      </a>
    </div>
  );
};

const BotInstallProcessForOfficialBot = () => {
  const { t } = useTranslation();
  return (
    <div className="my-5 d-flex flex-column align-items-center">
      <button type="button" className="btn btn-primary text-nowrap" onClick={() => window.open('https://slackbot-proxy.growi.org/', '_blank')}>
        {t('admin:slack_integration.accordion.install_now')}
        <i className="fa fa-external-link ml-2" aria-hidden="true" />
      </button>
      <a
        href={t('admin:slack_integration.docs_url.official_bot_setting')}
        target="_blank"
        rel="noopener noreferrer"
      >
        <p className="text-center mt-1">
          <small>
            {t('admin:slack_integration.accordion.how_to_install')}
            <i className="fa fa-external-link ml-2" aria-hidden="true" />
          </small>
        </p>
      </a>
    </div>
  );
};

const BotInstallProcessForCustomBotWithProxy = () => {
  const { t } = useTranslation();
  return (
    <div className="container w-75 py-5">
      <p>1. {t('admin:slack_integration.accordion.go-to-manage-distribution')}</p>
      <p>2. {t('admin:slack_integration.accordion.activate-public-distribution')}</p>
      <img src="/images/slack-integration/activate-public-dist.png" className="border border-light img-fluid mb-5" />
      <p>3. {t('admin:slack_integration.accordion.click-add-to-slack-button')}</p>
      <img src="/images/slack-integration/click-add-to-slack.png" className="border border-light img-fluid mb-5" />
      <p>4. {t('admin:slack_integration.accordion.click_allow')}</p>
      <img src="/images/slack-integration/slack-bot-install-your-app-transition-destination.png" className="border border-light img-fluid mb-5" />
      <p>5. {t('admin:slack_integration.accordion.install_complete_if_checked')}</p>
      <img src="/images/slack-integration/basicinfo-all-checked.png" className="border border-light img-fluid mb-5" />
      <p>6. {t('admin:slack_integration.accordion.invite_bot_to_channel')}</p>
      <img src="/images/slack-integration/slack-bot-install-to-workspace-joined-bot.png" className="border border-light img-fluid mb-1" />
      <img src="/images/slack-integration/slack-bot-install-your-app-introduction-to-channel.png" className="border border-light img-fluid" />
    </div>
  );
};

const RegisteringProxyUrlProcess = () => {
  const { t } = useTranslation();
  return (
    <div className="container w-75 py-5">
      <ol>
        <li>
          <p
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: t('admin:slack_integration.accordion.copy_proxy_url') }}
          />
          <p>
            <img className="border border-light img-fluid" src="/images/slack-integration/growi-register-sentence.png" />
          </p>
        </li>
        <li>
          <p
            // eslint-disable-next-line react/no-danger
            dangerouslySetInnerHTML={{ __html: t('admin:slack_integration.accordion.enter_proxy_url_and_update') }}
          />
          <p>
            <img className="border border-light img-fluid" src="/images/slack-integration/growi-set-proxy-url.png" />
          </p>
          <p className="text-danger">{t('admin:slack_integration.accordion.dont_need_update')}</p>
        </li>
      </ol>
    </div>
  );
};

// To get different messages for each copy happend, wrapping CopyToClipBoard and Tooltip together
const CustomCopyToClipBoard = (props) => {
  const { t } = useTranslation();
  const [tooltipOpen, setTooltipOpen] = useState(false);

  const showToolTip = useCallback(() => {
    setTooltipOpen(true);
    setTimeout(() => {
      setTooltipOpen(false);
    }, 1000);
  }, []);
  return (
    <>
      <CopyToClipboard text={props.textToBeCopied || ''} onCopy={showToolTip}>
        <div className="btn input-group-text" id="tooltipTarget">
          <i className="fa fa-clipboard mx-1" aria-hidden="true"></i>
        </div>
      </CopyToClipboard>
      <Tooltip target="tooltipTarget" fade={false} isOpen={tooltipOpen}>
        {t(props.message)}
      </Tooltip>
    </>
  );
};

const GeneratingTokensAndRegisteringProxyServiceProcess = withUnstatedContainers((props) => {
  const { t } = useTranslation();
  const { slackAppIntegrationId } = props;

  const regenerateTokensHandler = async() => {
    try {
      await apiv3Put(`/slack-integration-settings/slack-app-integrations/${slackAppIntegrationId}/regenerate-tokens`);
      if (props.onUpdateTokens != null) {
        props.onUpdateTokens();
      }
      toastSuccess(t('toaster.update_successed', { target: 'Token', ns: 'commons' }));
    }
    catch (err) {
      toastError(err);
      logger.error(err);
    }
  };

  return (
    <div className="py-4 px-5">
      <p className="font-weight-bold">1. {t('admin:slack_integration.accordion.generate_access_token')}</p>
      <div className="form-group row">
        <label className="text-left text-md-right col-md-3 col-form-label">Access Token Proxy to GROWI</label>
        <div className="col-md-6">
          <div className="input-group-prepend mx-1">
            <input className="form-control" type="text" value={props.tokenPtoG || ''} readOnly />
            <CustomCopyToClipBoard textToBeCopied={props.tokenPtoG || ''} message="admin:slack_integration.copied_to_clipboard"></CustomCopyToClipBoard>
          </div>
        </div>
      </div>
      <div className="form-group row">
        <label className="text-left text-md-right col-md-3 col-form-label">Access Token GROWI to Proxy</label>
        <div className="col-md-6">
          <div className="input-group-prepend mx-1">
            <input className="form-control" type="text" value={props.tokenGtoP || ''} readOnly />
            <CustomCopyToClipBoard textToBeCopied={props.tokenGtoP || ''} message="admin:slack_integration.copied_to_clipboard"></CustomCopyToClipBoard>
          </div>
        </div>
      </div>

      <div className="row my-3">
        <button
          type="button"
          className="btn btn-primary mx-auto"
          onClick={regenerateTokensHandler}
        >
          { t('admin:slack_integration.access_token_settings.regenerate') }
        </button>
      </div>
      <p className="font-weight-bold mt-5">2. {t('admin:slack_integration.accordion.register_for_growi_official_bot_proxy_service')}</p>
      <div className="d-flex flex-column align-items-center">
        <ol className="p-0">
          <li>
            <p
              className="ml-2"
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: t('admin:slack_integration.accordion.enter_growi_register_on_slack') }}
            />
          </li>
          <li>
            <p
              className="ml-2"
              // TODO: Add dynamic link
              // TODO: Add logo
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: t('admin:slack_integration.accordion.paste_growi_url') }}
            />
            <div className="input-group align-items-center pl-2 mb-3">
              <div className="input-group-prepend w-75">
                <input className="form-control" type="text" value={props.growiUrl} readOnly />
                <CustomCopyToClipBoard textToBeCopied={props.growiUrl} message="admin:slack_integration.copied_to_clipboard"></CustomCopyToClipBoard>
              </div>
            </div>

          </li>
          <li>
            <p
              className="ml-2"
              // eslint-disable-next-line react/no-danger
              dangerouslySetInnerHTML={{ __html: t('admin:slack_integration.accordion.enter_access_token_for_growi_and_proxy') }}
            />
          </li>
        </ol>
        <img className="mb-3 border border-light img-fluid" width={500} src="/images/slack-integration/growi-register-modal.png" />
      </div>
    </div>

  );
}, []);

const TestProcess = ({
  slackAppIntegrationId, onSubmitForm, onSubmitFormFailed, isLatestConnectionSuccess,
}) => {

  const { t } = useTranslation();
  const [testChannel, setTestChannel] = useState('');
  const [logsValue, setLogsValue] = useState('');
  const successMessage = 'Successfully sent to Slack workspace.';

  const submitForm = async(e) => {
    e.preventDefault();
    try {
      await apiv3Post(`/slack-integration-settings/slack-app-integrations/${slackAppIntegrationId}/relation-test`, { channel: testChannel });
      const newLogs = addLogs(logsValue, successMessage, null);
      setLogsValue(newLogs);

      if (onSubmitForm != null) {
        onSubmitForm();
      }
    }
    catch (error) {
      const newLogs = addLogs(logsValue, error[0].message, error[0].code);
      setLogsValue(newLogs);
      logger.error(error);
      if (onSubmitFormFailed != null) {
        onSubmitFormFailed();
      }
    }
  };

  return (
    <>
      <p className="text-center m-4">{t('admin:slack_integration.accordion.test_connection_by_pressing_button')}</p>
      <p className="text-center text-warning">
        <i className="icon-info">{t('admin:slack_integration.accordion.test_connection_only_public_channel')}</i>
      </p>
      <div className="d-flex justify-content-center">
        <form className="form-row justify-content-center" onSubmit={e => submitForm(e)}>
          <div className="input-group col-8">
            <div className="input-group-prepend">
              <span className="input-group-text" id="slack-channel-addon"><i className="fa fa-hashtag" /></span>
            </div>
            <input
              className="form-control"
              type="text"
              value={testChannel}
              placeholder="Slack Channel"
              onChange={e => setTestChannel(e.target.value)}
            />
          </div>
          <button
            type="submit"
            className="btn btn-info mx-3 font-weight-bold"
            disabled={testChannel.trim() === ''}
          >
            Test
          </button>
        </form>
      </div>
      <MessageBasedOnConnection isLatestConnectionSuccess={isLatestConnectionSuccess} logsValue={logsValue} />
      <form>
        <div className="row my-3 justify-content-center">
          <div className="form-group slack-connection-log col-md-4">
            <label className="mb-1"><p className="border-info slack-connection-log-title pl-2 m-0">Logs</p></label>
            <textarea
              className="form-control card border-info slack-connection-log-body rounded-lg"
              rows="5"
              value={logsValue}
              readOnly
            />
          </div>
        </div>
      </form>
    </>
  );
};


const WithProxyAccordions = (props) => {
  const { t } = useTranslation();
  const { data: siteUrl } = useSiteUrl();
  const [isLatestConnectionSuccess, setIsLatestConnectionSuccess] = useState(false);

  const submitForm = () => {
    setIsLatestConnectionSuccess(true);
    if (props.onSubmitForm != null) {
      props.onSubmitForm();
    }
  };
  const submitFormFailed = () => {
    setIsLatestConnectionSuccess(false);
  };


  const officialBotIntegrationProcedure = {
    '①': {
      title: 'install_bot_to_slack',
      content: <BotInstallProcessForOfficialBot />,
    },
    '②': {
      title: 'register_for_growi_official_bot_proxy_service',
      content: <GeneratingTokensAndRegisteringProxyServiceProcess
        growiUrl={siteUrl}
        slackAppIntegrationId={props.slackAppIntegrationId}
        tokenPtoG={props.tokenPtoG}
        tokenGtoP={props.tokenGtoP}
        onUpdateTokens={props.onUpdateTokens}
      />,
    },
    '③': {
      title: 'manage_permission',
      content: <ManageCommandsProcess
        slackAppIntegrationId={props.slackAppIntegrationId}
        permissionsForBroadcastUseCommands={props.permissionsForBroadcastUseCommands}
        permissionsForSingleUseCommands={props.permissionsForSingleUseCommands}
        permissionsForSlackEventActions={props.permissionsForSlackEventActions}
      />,
    },
    '④': {
      title: 'test_connection',
      content: <TestProcess
        slackAppIntegrationId={props.slackAppIntegrationId}
        onSubmitForm={submitForm}
        onSubmitFormFailed={submitFormFailed}
        isLatestConnectionSuccess={isLatestConnectionSuccess}
      />,
    },
  };

  const CustomBotIntegrationProcedure = {
    '①': {
      title: 'create_bot',
      content: <BotCreateProcess />,
    },
    '②': {
      title: 'install_bot_to_slack',
      content: <BotInstallProcessForCustomBotWithProxy />,
    },
    '③': {
      title: 'register_for_growi_custom_bot_proxy',
      content: <GeneratingTokensAndRegisteringProxyServiceProcess
        growiUrl={siteUrl}
        slackAppIntegrationId={props.slackAppIntegrationId}
        tokenPtoG={props.tokenPtoG}
        tokenGtoP={props.tokenGtoP}
        onUpdateTokens={props.onUpdateTokens}
      />,
    },
    '④': {
      title: 'set_proxy_url_on_growi',
      content: <RegisteringProxyUrlProcess />,
    },
    '⑤': {
      title: 'manage_permission',
      content: <ManageCommandsProcess
        slackAppIntegrationId={props.slackAppIntegrationId}
        permissionsForBroadcastUseCommands={props.permissionsForBroadcastUseCommands}
        permissionsForSingleUseCommands={props.permissionsForSingleUseCommands}
        permissionsForSlackEventActions={props.permissionsForSlackEventActions}
      />,
    },
    '⑥': {
      title: 'test_connection',
      content: <TestProcess
        slackAppIntegrationId={props.slackAppIntegrationId}
        onSubmitForm={submitForm}
        onSubmitFormFailed={submitFormFailed}
        isLatestConnectionSuccess={isLatestConnectionSuccess}
      />,
    },
  };

  const integrationProcedureMapping = props.botType === SlackbotType.OFFICIAL ? officialBotIntegrationProcedure : CustomBotIntegrationProcedure;

  return (
    <div
      className="card border-0 rounded-lg shadow overflow-hidden"
    >
      {Object.entries(integrationProcedureMapping).map(([key, value]) => {
        return (
          <Accordion
            title={(
              <>
                <span className="mr-2">{key}</span>
                {t(`admin:slack_integration.accordion.${value.title}`)}
                {value.title === 'test_connection' && isLatestConnectionSuccess && <i className="ml-3 text-success fa fa-check"></i>}
              </>
            )}
            key={key}
          >
            {value.content}
          </Accordion>
        );
      })}
    </div>
  );
};


/**
 * Wrapper component for using unstated
 */
WithProxyAccordions.propTypes = {
  botType: PropTypes.oneOf(Object.values(SlackbotType)).isRequired,
  slackAppIntegrationId: PropTypes.string.isRequired,
  tokenPtoG: PropTypes.string,
  tokenGtoP: PropTypes.string,
  permissionsForBroadcastUseCommands: PropTypes.object.isRequired,
  permissionsForSingleUseCommands: PropTypes.object.isRequired,
  permissionsForSlackEventActions: PropTypes.object.isRequired,
};

export default WithProxyAccordions;
