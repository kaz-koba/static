import { Root, Element, Text } from 'hast';
import rehypeRewrite from 'rehype-rewrite';
import { Plugin } from 'unified';


/**
 * This method returns ['foo', 'bar', 'foo']
 *  when the arguments are { keyword: 'foo', value: 'foobarfoo' }
 * @param keyword
 * @param value
 * @returns
 */
function splitWithKeyword(keyword: string, value: string): string[] {
  if (value.length === 0) {
    return [];
  }

  let cursorStart = 0;
  let cursorEnd = 0;

  const splitted: string[] = [];

  do {
    cursorEnd = value.indexOf(keyword, cursorStart);

    // not found
    if (cursorEnd === -1) {
      cursorEnd = value.length;
    }
    // keyword is found
    else if (cursorEnd === cursorStart) {
      cursorEnd += keyword.length;
    }

    splitted.push(value.slice(cursorStart, cursorEnd));
    cursorStart = cursorEnd;
  } while (cursorStart < value.length);

  return splitted;
}

function wrapWithEm(textElement: Text): Element {
  return {
    type: 'element',
    tagName: 'em',
    properties: {
      className: 'highlighted-keyword',
    },
    children: [textElement],
  };
}

function highlight(keyword: string, node: Text, index: number, parent: Root | Element): void {
  if (node.value.includes(keyword)) {
    const splitted = splitWithKeyword(keyword, node.value);

    parent.children[index] = {
      type: 'element',
      tagName: 'span',
      properties: {},
      children: splitted.map((text) => {
        return text === keyword
          ? wrapWithEm({ type: 'text', value: keyword })
          : { type: 'text', value: text };
      }),
    };
  }
}


export type KeywordHighlighterPluginParams = {
  keywords?: string | string[],
}

export const rehypePlugin: Plugin<[KeywordHighlighterPluginParams]> = (options) => {
  if (options?.keywords == null) {
    return node => node;
  }

  const keywords = (typeof options.keywords === 'string') ? [options.keywords] : options.keywords;

  // return rehype-rewrite with hithlighter
  return rehypeRewrite.bind(this)({
    rewrite: (node, index, parent) => {
      if (parent != null && index != null && node.type === 'text') {
        keywords.forEach(keyword => highlight(keyword, node, index, parent));
      }
    },
  });
};
