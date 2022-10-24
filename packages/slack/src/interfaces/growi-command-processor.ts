import { AuthorizeResult } from '@slack/oauth';

import { GrowiCommand } from './growi-command';

export interface GrowiCommandProcessor<ProcessCommandContext = {[key: string]: string}> {
  shouldHandleCommand(growiCommand?: GrowiCommand): boolean;

  processCommand(growiCommand: GrowiCommand, authorizeResult: AuthorizeResult, context?: ProcessCommandContext): Promise<void>
}
