import React, { FC, forwardRef, useCallback } from 'react';

import { addDays, format } from 'date-fns';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';


type CustomInputProps = {
  value?: string
  onChange?: () => void
  onFocus?: () => void
}

const CustomInput = forwardRef<HTMLInputElement, CustomInputProps>((props: CustomInputProps, ref) => {
  const dateFormat = 'MM/dd/yyyy';
  const date = new Date();
  const placeholder = `${format(date, dateFormat)} - ${format(addDays(date, 1), dateFormat)}`;

  return (
    <div className="input-group admin-audit-log">
      <div className="input-group-prepend">
        <span className="input-group-text">
          <i className="fa fa-fw fa-calendar" />
        </span>
      </div>
      <input
        ref={ref}
        type="text"
        value={props?.value}
        onFocus={props?.onFocus}
        onChange={props?.onChange}
        placeholder={placeholder}
        className="form-control date-range-picker"
        aria-describedby="basic-addon1"
      />
    </div>
  );
});

CustomInput.displayName = 'CustomInput';

type DateRangePickerProps = {
  startDate: Date | null
  endDate: Date | null
  onChange: (dateList: Date[] | null[]) => void
}

export const DateRangePicker: FC<DateRangePickerProps> = (props: DateRangePickerProps) => {
  const { startDate, endDate, onChange } = props;

  const changeHandler = useCallback((dateList: Date[] | null[]) => {
    if (onChange != null) {
      const [start, end] = dateList;
      const isSameTime = (start != null && end != null) && (start.getTime() === end.getTime());
      if (isSameTime) {
        onChange([null, null]);
      }
      else {
        onChange(dateList);
      }
    }
  }, [onChange]);

  return (
    <div className="btn-group mr-2">
      <DatePicker
        selectsRange
        startDate={startDate}
        endDate={endDate}
        onChange={changeHandler}
        customInput={<CustomInput />}
      />
    </div>
  );
};
