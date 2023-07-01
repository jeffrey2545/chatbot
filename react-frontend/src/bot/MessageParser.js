import React from 'react';

const MessageParser = (props) => {
  const {children, actions} = props;

  const parse = (message) => {
    if (message.includes('hello')) {
      actions.handleHello();
    }
    if (message.includes('dog')) {
      actions.handleDog();
    }
    if (message.includes('client')) {
      actions.handleClientMessage();
    }
    if (message.includes('custom')) {
      actions.handleCustomMessage();
    }
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