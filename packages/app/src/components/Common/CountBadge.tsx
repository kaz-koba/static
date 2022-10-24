import React, { FC } from 'react';

type CountProps = {
  count?: number,
  offset?: number,
}

const CountBadge: FC<CountProps> = (props:CountProps) => {
  const { count, offset = 0 } = props;


  return (
    <span className="grw-count-badge px-2 badge badge-pill badge-light">
      { count == null && <span className="text-muted">―</span> }
      { count != null && count + offset }
    </span>
  );
};

export default CountBadge;
