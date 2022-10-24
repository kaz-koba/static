import React, { useState } from 'react';
import { Collapse } from 'reactstrap';
import PropTypes from 'prop-types';

const Accordion = (props) => {
  const [isOpen, setIsOpen] = useState(props.isOpenDefault);
  return (
    <div className="card border-0 rounded-lg mb-0">
      <div
        className="card-header font-weight-normal py-3 d-flex justify-content-between"
        role="button"
        onClick={() => setIsOpen(prevState => !prevState)}
      >
        <p className="mb-0">{props.title}</p>
        {isOpen
          ? <i className="fa fa-chevron-up" />
          : <i className="fa fa-chevron-down" />
        }
      </div>
      <Collapse isOpen={isOpen}>
        <div className="card-body">
          {props.children}
        </div>
      </Collapse>
    </div>
  );
};

Accordion.propTypes = {
  title: PropTypes.node.isRequired,
  children: PropTypes.node.isRequired,
  isOpenDefault: PropTypes.bool,
};

Accordion.defaultProps = {
  isOpenDefault: false,
};

export default Accordion;
