import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8080/v1",
  apiKey: "neuromem-local",
});

const response = await client.chat.completions.create({
  model: "your-upstream-model",
  messages: [
    { role: "system", content: "You are a helpful engineering assistant." },
    { role: "user", content: "Find the most relevant project memory for the current bug." },
  ],
  extra_body: {
    neuromem: {
      session_id: "demo-project",
      top_k: 5,
      store_messages: true,
    },
  },
});

console.log(response.choices[0].message.content);
