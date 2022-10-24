import loggerFactory from '~/utils/logger';

const {
  markdownHeaderBlock, inputSectionBlock, inputBlock, actionsBlock, buttonElement,
} = require('@growi/slack');
const { SlackCommandHandlerError } = require('../../models/vo/slack-command-handler-error');

const logger = loggerFactory('growi:service:SlackCommandHandler:note');

module.exports = (crowi) => {
  const CreatePageService = require('./create-page-service');
  const createPageService = new CreatePageService(crowi);
  const BaseSlackCommandHandler = require('./slack-command-handler');
  const handler = new BaseSlackCommandHandler();
  const conversationsSelectElement = {
    action_id: 'conversation',
    type: 'conversations_select',
    default_to_current_conversation: true,
  };

  handler.handleCommand = async(growiCommand, client, body, respondUtil) => {
    await respondUtil.respond({
      text: 'Take a note on GROWI',
      blocks: [
        markdownHeaderBlock('Take a note on GROWI'),
        inputBlock(conversationsSelectElement, 'conversation', 'Channel name to display in the page to be created'),
        inputSectionBlock('path', 'Page path', 'path_input', false, '/path'),
        inputSectionBlock('contents', 'Contents', 'contents_input', true, 'Input with Markdown...'),
        actionsBlock(
          buttonElement({ text: 'Cancel', actionId: 'note:cancel' }),
          buttonElement({ text: 'Create page', actionId: 'note:createPage', style: 'primary' }),
        ),

      ],
    });
  };

  handler.cancel = async function(client, interactionPayload, interactionPayloadAccessor, respondUtil) {
    await respondUtil.deleteOriginal();
  };

  handler.handleInteractions = async function(client, interactionPayload, interactionPayloadAccessor, handlerMethodName, respondUtil) {
    await this[handlerMethodName](client, interactionPayload, interactionPayloadAccessor, respondUtil);
  };

  handler.createPage = async function(client, interactionPayload, interactionPayloadAccessor, respondUtil) {
    const path = interactionPayloadAccessor.getStateValues()?.path.path_input.value;
    const contentsBody = interactionPayloadAccessor.getStateValues()?.contents.contents_input.value;
    if (path == null || contentsBody == null) {
      throw new SlackCommandHandlerError('All parameters are required.');
    }
    await createPageService.createPageInGrowi(interactionPayloadAccessor, path, contentsBody, respondUtil);
    await respondUtil.deleteOriginal();
  };

  return handler;
};
