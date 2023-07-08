import React from 'react';

const MessageParser = (props) => {
  const {children, actions} = props;

  const parse = (message) => {
    actions.handleByOpenAI(message);
  };

  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          parse: parse,
          actions: {},
        });
      })}
    </div>
  );
};

export default MessageParser;