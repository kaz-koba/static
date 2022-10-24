import nodePath from 'path';

import escapeStringRegexp from 'escape-string-regexp';

import { isValidObjectId } from './objectid-utils';
import { addTrailingSlash } from './path-utils';

/**
 * Whether path is the top page
 * @param path
 */
export const isTopPage = (path: string): boolean => {
  return path === '/';
};

/**
 * Whether path is the top page of users
 * @param path
 */
export const isUsersTopPage = (path: string): boolean => {
  return path === '/user';
};

/**
 * Whether the path is permalink
 * @param path
 */
export const isPermalink = (path: string): boolean => {
  const pageIdStr = path.substring(1);
  return isValidObjectId(pageIdStr);
};

/**
 * Whether path is user's home page
 * @param path
 */
export const isUsersHomePage = (path: string): boolean => {
  // https://regex101.com/r/utVQct/1
  if (path.match(/^\/user\/[^/]+$/)) {
    return true;
  }
  return false;
};

/**
 * Whether path is the protected pages for systems
 * @param path
 */
export const isUsersProtectedPages = (path: string): boolean => {
  return isUsersTopPage(path) || isUsersHomePage(path);
};

/**
 * Whether path is movable
 * @param path
 */
export const isMovablePage = (path: string): boolean => {
  return !isTopPage(path) && !isUsersProtectedPages(path);
};

/**
 * Whether path belongs to the user page
 * @param path
 */
export const isUserPage = (path: string): boolean => {
  // https://regex101.com/r/MwifLR/1
  if (path.match(/^\/user\/.*?$/)) {
    return true;
  }

  return false;
};

/**
 * Whether path is the top page of users
 * @param path
 */
export const isTrashTopPage = (path: string): boolean => {
  return path === '/trash';
};

/**
 * Whether path belongs to the trash page
 * @param path
 */
export const isTrashPage = (path: string): boolean => {
  // https://regex101.com/r/BSDdRr/1
  if (path.match(/^\/trash(\/.*)?$/)) {
    return true;
  }

  return false;
};

/**
 * Whether path belongs to the shared page
 * @param path
 */
export const isSharedPage = (path: string): boolean => {
  // https://regex101.com/r/ZjdOiB/1
  if (path.match(/^\/share(\/.*)?$/)) {
    return true;
  }

  return false;
};

const restrictedPatternsToCreate: Array<RegExp> = [
  /\^|\$|\*|\+|#|%|\?/,
  /^\/-\/.*/,
  /^\/_r\/.*/,
  /^\/_apix?(\/.*)?/,
  /^\/?https?:\/\/.+$/, // avoid miss in renaming
  /\/{2,}/, // avoid miss in renaming
  /\s+\/\s+/, // avoid miss in renaming
  /.+\/edit$/,
  /.+\.md$/,
  /^(\.\.)$/, // see: https://github.com/weseek/growi/issues/3582
  /(\/\.\.)\/?/, // see: https://github.com/weseek/growi/issues/3582
  /^\/(_search|_private-legacy-pages)(\/.*|$)/,
  /^\/(installer|register|login|logout|admin|me|files|trash|paste|comments|tags|share)(\/.*|$)/,
  /^\/user\/[^/]+$/, // see: https://regex101.com/r/utVQct/1
];
export const isCreatablePage = (path: string): boolean => {
  return !restrictedPatternsToCreate.some(pattern => path.match(pattern));
};

/**
 * return user path
 * @param user
 */
// eslint-disable-next-line @typescript-eslint/explicit-module-boundary-types
export const userPageRoot = (user: any): string => {
  if (!user || !user.username) {
    return '';
  }
  return `/user/${user.username}`;
};

/**
 * return user path
 * @param parentPath
 * @param childPath
 * @param newPath
 */
export const convertToNewAffiliationPath = (oldPath: string, newPath: string, childPath: string): string => {
  if (newPath == null) {
    throw new Error('Please input the new page path');
  }
  const pathRegExp = new RegExp(`^${escapeStringRegexp(oldPath)}`, 'i');
  return childPath.replace(pathRegExp, newPath);
};

/**
 * Encode SPACE and IDEOGRAPHIC SPACE
 * @param {string} path
 * @returns {string}
 */
export const encodeSpaces = (path?:string): string | undefined => {
  if (path == null) {
    return undefined;
  }

  // Encode SPACE and IDEOGRAPHIC SPACE
  return path.replace(/ /g, '%20').replace(/\u3000/g, '%E3%80%80');
};

/**
 * Generate editor path
 * @param {string} paths
 * @returns {string}
 */
export const generateEditorPath = (...paths: string[]): string => {
  const joinedPath = [...paths].join('/');

  if (!isCreatablePage(joinedPath)) {
    throw new Error('Invalid characters on path');
  }

  try {
    const url = new URL(joinedPath, 'https://dummy');
    return `${url.pathname}#edit`;
  }
  catch (err) {
    throw new Error('Invalid path format');
  }
};

/**
 * returns ancestors paths
 * @param {string} path
 * @param {string[]} ancestorPaths
 * @returns {string[]}
 */
export const collectAncestorPaths = (path: string, ancestorPaths: string[] = []): string[] => {
  if (isTopPage(path)) return ancestorPaths;

  const parentPath = nodePath.dirname(path);
  ancestorPaths.push(parentPath);
  return collectAncestorPaths(parentPath, ancestorPaths);
};

/**
 * return paths without duplicate area of regexp /^${path}\/.+/i
 * ex. expect(omitDuplicateAreaPathFromPaths(['/A', '/A/B', '/A/B/C'])).toStrictEqual(['/A'])
 * @param paths paths to be tested
 * @returns omitted paths
 */
export const omitDuplicateAreaPathFromPaths = (paths: string[]): string[] => {
  const uniquePaths = Array.from(new Set(paths));
  return uniquePaths.filter((path) => {
    const isDuplicate = uniquePaths.filter(p => (new RegExp(`^${p}\\/.+`, 'i')).test(path)).length > 0;

    return !isDuplicate;
  });
};

/**
 * return pages with path without duplicate area of regexp /^${path}\/.+/i
 * if the pages' path are the same, it will NOT omit any of them since the other attributes will not be the same
 * @param paths paths to be tested
 * @returns omitted paths
 */
export const omitDuplicateAreaPageFromPages = (pages: any[]): any[] => {
  return pages.filter((page) => {
    const isDuplicate = pages.some(p => (new RegExp(`^${p.path}\\/.+`, 'i')).test(page.path));

    return !isDuplicate;
  });
};

/**
 * Check if the area of either path1 or path2 includes the area of the other path
 * The area of path is the same as /^\/hoge\//i
 * @param pathToTest string
 * @param pathToBeTested string
 * @returns boolean
 */
export const isEitherOfPathAreaOverlap = (path1: string, path2: string): boolean => {
  if (path1 === path2) {
    return true;
  }

  const path1WithSlash = addTrailingSlash(path1);
  const path2WithSlash = addTrailingSlash(path2);

  const path1Area = new RegExp(`^${escapeStringRegexp(path1WithSlash)}`, 'i');
  const path2Area = new RegExp(`^${escapeStringRegexp(path2WithSlash)}`, 'i');

  if (path1Area.test(path2) || path2Area.test(path1)) {
    return true;
  }

  return false;
};

/**
 * Check if the area of pathToTest includes the area of pathToBeTested
 * The area of path is the same as /^\/hoge\//i
 * @param pathToTest string
 * @param pathToBeTested string
 * @returns boolean
 */
export const isPathAreaOverlap = (pathToTest: string, pathToBeTested: string): boolean => {
  if (pathToTest === pathToBeTested) {
    return true;
  }

  const pathWithSlash = addTrailingSlash(pathToTest);

  const pathAreaToTest = new RegExp(`^${escapeStringRegexp(pathWithSlash)}`, 'i');
  if (pathAreaToTest.test(pathToBeTested)) {
    return true;
  }

  return false;
};

/**
 * Determine whether can move by fromPath and toPath
 * @param fromPath string
 * @param toPath string
 * @returns boolean
 */
export const canMoveByPath = (fromPath: string, toPath: string): boolean => {
  return !isPathAreaOverlap(fromPath, toPath);
};

/**
 * check if string has '/' in it
 */
export const hasSlash = (str: string): boolean => {
  return str.includes('/');
};

/**
 * Generate RegExp instance for one level lower path
 */
export const generateChildrenRegExp = (path: string): RegExp => {
  // https://regex101.com/r/laJGzj/1
  // ex. /any_level1
  if (isTopPage(path)) return new RegExp(/^\/[^/]+$/);

  // https://regex101.com/r/mrDJrx/1
  // ex. /parent/any_child OR /any_level1
  return new RegExp(`^${path}(\\/[^/]+)\\/?$`);
};
