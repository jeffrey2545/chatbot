import Chatbot from 'react-chatbot-kit'
import 'react-chatbot-kit/build/main.css'
import config from './bot/config';
import MessageParser from './bot/MessageParser.js';
import ActionProvider from './bot/ActionProvider.jsx';

const App = (props) => {
  return (
    <div className="App">
      <Chatbot
        config={config}
        messageParser={MessageParser}
        actionProvider={ActionProvider}
      />
    </div>
  );
}

export default App;
