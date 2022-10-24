import * as pathUtils from '../utils/path-utils';

// https://regex101.com/r/BahpKX/2
const PATTERN_INCLUDE_DATE = /^(.+\/[^/]+)\/(\d{4}|\d{4}\/\d{2}|\d{4}\/\d{2}\/\d{2})$/;

export class DevidedPagePath {

  constructor(path, skipNormalize = false, evalDatePath = false) {

    this.isRoot = false;
    this.isFormerRoot = false;
    this.former = null;
    this.latter = null;

    // root
    if (path == null || path === '' || path === '/') {
      this.isRoot = true;
      this.latter = '/';
      return;
    }

    const pagePath = skipNormalize ? path : pathUtils.normalizePath(path);
    this.latter = pagePath;

    // evaluate date path
    if (evalDatePath) {
      const matchDate = pagePath.match(PATTERN_INCLUDE_DATE);
      if (matchDate != null) {
        this.former = matchDate[1];
        this.latter = matchDate[2];
        return;
      }
    }

    let PATTERN_DEFAULT = /^((.*)\/)?(.+)$/; // this will not ignore html end tags https://regex101.com/r/jpZwIe/1
    try { // for non-chrome browsers
      // eslint-disable-next-line regex/invalid
      PATTERN_DEFAULT = new RegExp('^((.*)(?<!<)\\/)?(.+)$'); // https://regex101.com/r/HJNvMW/1
    }
    catch (err) {
      // lookbehind regex is not supported on non-chrome browsers
    }

    const matchDefault = pagePath.match(PATTERN_DEFAULT);
    if (matchDefault != null) {
      this.isFormerRoot = matchDefault[1] === '/';
      this.former = matchDefault[2];
      this.latter = matchDefault[3];
    }
  }

}
