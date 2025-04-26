import React, { useState } from "react";

function App() {
  const [message, setMessage] = useState("");

  const sendData = async () => {
    const response = await fetch("http://localhost:5000/api/echo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "Hello from React!" }),
    });
    const data = await response.json();
    setMessage(data.you_sent.message);
  };

  return (
    <div className="App">
      <h1>React + Flask Boilerplate</h1>
      <button onClick={sendData}>Send Message</button>
      {message && <p>Response: {message}</p>}
    </div>
  );
}

export default App;
