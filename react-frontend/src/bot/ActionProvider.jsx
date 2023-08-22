import React, { useEffect, useState } from 'react';
import { createChatBotMessage, createClientMessage, createCustomMessage } from 'react-chatbot-kit';

const ActionProvider = (props) => {
  const {createChatBotMessage, setState, children} = props;

  const handleByOpenAI = async (message) => {
    const encodedMessage = encodeURIComponent(message);
    const response = await fetch(`http://127.0.0.1:5000/langchain/chat/agent/${encodedMessage}`);
    const data = await response.json();
    const responseMessage = createChatBotMessage(data);

    setState((prev) => ({
      ...prev,
      messages: [...prev.messages, responseMessage],
    }));
  };

  return (
    <div>
      {React.Children.map(children, (child) => {
        return React.cloneElement(child, {
          actions: {
            handleByOpenAI
          },
        });
      })}
    </div>
  );
};

export default ActionProvider;