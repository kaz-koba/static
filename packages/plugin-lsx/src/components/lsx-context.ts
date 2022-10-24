import { customTagUtils, ParseRangeResult } from '@growi/core';

const { OptionParser } = customTagUtils;


export class LsxContext {

  pagePath: string;

  options?: Record<string, string|undefined>;

  constructor(pagePath: string, options: Record<string, string|undefined>) {
    this.pagePath = pagePath;

    // remove undefined keys
    Object.keys(options).forEach(key => options[key] === undefined && delete options[key]);

    this.options = options;
  }

  getOptDepth(): ParseRangeResult | null {
    if (this.options?.depth == null) {
      return null;
    }
    return OptionParser.parseRange(this.options.depth);
  }

  getStringifiedAttributes(separator = ', '): string {
    const attributeStrs = [`prefix=${this.pagePath}`];
    if (this.options != null) {
      const optionEntries = Object.entries(this.options).sort();
      attributeStrs.push(
        ...optionEntries.map(([key, val]) => `${key}=${val || 'true'}`),
      );
    }

    return attributeStrs.join(separator);
  }

  /**
   * for printing errors
   * @returns
   */
  toString(): string {
    return `$lsx(${this.getStringifiedAttributes()})`;
  }

}
