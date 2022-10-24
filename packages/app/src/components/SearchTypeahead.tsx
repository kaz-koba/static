import React, {
  FC, ForwardRefRenderFunction, forwardRef, useImperativeHandle,
  KeyboardEvent, useCallback, useRef, useState, MouseEvent, useEffect,
} from 'react';

import { UserPicture, PageListMeta, PagePathLabel } from '@growi/ui';
import { AsyncTypeahead, Menu, MenuItem } from 'react-bootstrap-typeahead';

import { IFocusable } from '~/client/interfaces/focusable';
import { TypeaheadProps } from '~/client/interfaces/react-bootstrap-typeahead';
import { IPageWithSearchMeta } from '~/interfaces/search';
import { useSWRxSearch } from '~/stores/search';


import styles from './SearchTypeahead.module.scss';


type ResetFormButtonProps = {
  input?: string,
  onReset: (e: MouseEvent<HTMLButtonElement>) => void,
}

const ResetFormButton: FC<ResetFormButtonProps> = (props: ResetFormButtonProps) => {
  const { input, onReset } = props;

  const isHidden = input == null || input.length === 0;

  return isHidden ? (
    <span />
  ) : (
    <button type="button" className="btn btn-outline-secondary search-clear text-muted border-0" onMouseDown={onReset}>
      <i className="icon-close" />
    </button>
  );
};


type Props = TypeaheadProps & {
  onSearchError?: (err: Error) => void,
  onSubmit?: (input: string) => void,
  keywordOnInit?: string,
  disableIncrementalSearch?: boolean,
  helpElement?: React.ReactNode,
};

// see https://github.com/ericgio/react-bootstrap-typeahead/issues/266#issuecomment-414987723
type TypeaheadInstance = {
  clear: () => void,
  focus: () => void,
  toggleMenu: () => void,
  state: { selected: IPageWithSearchMeta[] }
}

const SearchTypeahead: ForwardRefRenderFunction<IFocusable, Props> = (props: Props, ref) => {
  const {
    onSearchError, onSearch, onInputChange, onChange, onSubmit,
    inputProps, keywordOnInit, disableIncrementalSearch, helpElement,
    onBlur, onFocus,
  } = props;

  const [input, setInput] = useState(keywordOnInit);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [isForcused, setFocused] = useState(false);

  const { data: searchResult, error: searchError } = useSWRxSearch(
    disableIncrementalSearch ? null : searchKeyword,
    null,
    { limit: 10 },
  );

  const typeaheadRef = useRef<TypeaheadInstance>(null);

  const focusToTypeahead = () => {
    const instance = typeaheadRef.current;
    if (instance != null) {
      instance.focus();
    }
  };

  const clearTypeahead = () => {
    const instance = typeaheadRef.current;
    if (instance != null) {
      instance.clear();
    }
  };

  // publish focus()
  useImperativeHandle(ref, () => ({
    focus: focusToTypeahead,
  }));

  const resetForm = useCallback((e: MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();

    setInput('');
    setSearchKeyword('');

    clearTypeahead();
    focusToTypeahead();

    if (onSearch != null) {
      onSearch('');
    }
  }, [onSearch]);

  const searchHandler = useCallback((text: string) => {
    setSearchKeyword(text);

    if (onSearch != null) {
      onSearch(text);
    }
  }, [onSearch]);

  const inputChangeHandler = useCallback((text: string) => {
    setInput(text);

    if (onInputChange != null) {
      onInputChange(text);
    }
  }, [onInputChange]);

  /* -------------------------------------------------------------------------------------------------------
   *
   * Dirty hack for https://github.com/ericgio/react-bootstrap-typeahead/issues/492 -- 2022.03.22 Yuki Takei
   *
   * 1. Schedule to submit with delay when Enter key downed
   * 2. Fire onChange and cancel the schedule to submit if change event occured
   * 3. Fire onSubmit if the schedule is not canceled
   *
   */
  const DELAY_FOR_SUBMISSION = 100;
  const timeoutIdRef = useRef<NodeJS.Timeout>();

  const changeHandler = useCallback((selectedItems: IPageWithSearchMeta[]) => {
    // cancel schedule to submit
    if (timeoutIdRef.current != null) {
      clearTimeout(timeoutIdRef.current);
    }

    if (selectedItems.length > 0) {
      setInput(selectedItems[0].data.path);

      if (onChange != null) {
        onChange(selectedItems);
      }
    }
  }, [onChange]);

  const keyDownHandler = useCallback((event: KeyboardEvent) => {
    if (event.keyCode === 13) { // Enter key
      if (onSubmit != null && input != null && input.length > 0) {
        // schedule to submit with 100ms delay
        timeoutIdRef.current = setTimeout(() => onSubmit(input), DELAY_FOR_SUBMISSION);
      }
    }
  }, [input, onSubmit]);
  /*
   * -------------------------------------------------------------------------------------------------------
   */

  useEffect(() => {
    if (onSearchError != null && searchError != null) {
      onSearchError(searchError);
    }
  }, [onSearchError, searchError]);

  const labelKey = useCallback((option?: IPageWithSearchMeta) => {
    return option?.data.path ?? '';
  }, []);

  const renderMenu = useCallback((options: IPageWithSearchMeta[], menuProps) => {
    if (!isForcused) {
      return <></>;
    }

    const isEmptyInput = input == null || input.length === 0;
    if (isEmptyInput) {
      if (helpElement == null) {
        return <></>;
      }

      return (
        <Menu {...menuProps}>
          <div className="p-3">
            {helpElement}
          </div>
        </Menu>
      );
    }

    if (disableIncrementalSearch) {
      return <></>;
    }

    return (
      <Menu {...menuProps}>
        {options.map((pageWithMeta, index) => (
          <MenuItem key={pageWithMeta.data._id} option={pageWithMeta} position={index}>
            <span>
              <UserPicture user={pageWithMeta.data.lastUpdateUser} size="sm" noLink />
              <span className="ml-1 mr-2 text-break text-wrap"><PagePathLabel path={pageWithMeta.data.path} /></span>
              <PageListMeta page={pageWithMeta.data} />
            </span>
          </MenuItem>
        ))}
      </Menu>
    );
  }, [disableIncrementalSearch, helpElement, input, isForcused]);

  const isLoading = searchResult !== undefined && searchError == null;
  const isOpenAlways = helpElement != null;

  return (
    <div className={`search-typeahead ${styles['search-typeahead']}`}>
      <AsyncTypeahead
        {...props}
        id="search-typeahead-asynctypeahead"
        ref={typeaheadRef}
        delay={400}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        inputProps={{ autoComplete: 'off', ...(inputProps as any ?? {}) }}
        isLoading={isLoading}
        labelKey={labelKey}
        defaultInputValue={keywordOnInit}
        options={searchResult?.data} // Search result (Some page names)
        align="left"
        open={isOpenAlways || undefined}
        renderMenu={renderMenu}
        autoFocus={props.autoFocus}
        onChange={changeHandler}
        onSearch={searchHandler}
        onInputChange={inputChangeHandler}
        onKeyDown={keyDownHandler}
        onBlur={() => {
          setFocused(false);
          if (onBlur != null) {
            onBlur();
          }
        }}
        onFocus={() => {
          setFocused(true);
          if (onFocus != null) {
            onFocus();
          }
        }}
      />
      <ResetFormButton
        input={input}
        onReset={resetForm}
      />
    </div>
  );
};

const ForwardedSearchTypeahead = forwardRef(SearchTypeahead);

ForwardedSearchTypeahead.defaultProps = {
  placeholder: '',
  keywordOnInit: '',
  autoFocus: false,
};

export default ForwardedSearchTypeahead;
