import React, { useEffect, useState } from 'react';
import { createChatBotMessage, createClientMessage, createCustomMessage } from 'react-chatbot-kit';

const ActionProvider = (props) => {
  const {createChatBotMessage, setState, children} = props;

  const handleHello = () => {
    const botMessage = createChatBotMessage('Hello. Nice to meet you.');

    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, botMessage],
    }));
  };

  const handleDog = () => {
    const botMessage = createChatBotMessage(
      "Here's a nice dog picture for you!",
      {
        widget: 'dogPicture',
      }
    );

    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, botMessage],
    }));
  };

  const handleClientMessage = () => {
    const message = createClientMessage('Hello client message!');
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
  };

  const handleCustomMessage = () => {
    // 1st. argument is the text value, 2nd. argument is the name of the registered custom message.
    const message = createCustomMessage('value to input', 'custom');
    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, message],
    }));
  };

  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          actions: {
            handleHello,
            handleDog,
            handleClientMessage,
            handleCustomMessage,
          },
        });
      })}
    </div>
  );
};

export default ActionProvider;