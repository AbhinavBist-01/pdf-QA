import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function query(userQuery) {
  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: "pdf-qa",
    index: new CloudClient({
      apiKey: process.env.CHROMADB_API_KEY,
      tenant: "ae7af065-af71-456d-8c9c-3127e359d578",
      database: "pdf-qa",
    }),
  });

  const vectorStoreRetriever = vectorStore.asRetriever({ k: 5 });

  const results = await vectorStoreRetriever.invoke(userQuery);

  const SYSTEM_PROMPT = `
    You are an expert in answering user query based on the provided context about document.
    Do not answer anything beyond what is provided.

    Always also answer the user concisely and state which page number that content is available on and the name/source of the document.

    User Documents:
    ${results.map((e) => JSON.stringify({ bookName: e.metadata?.source, pageContent: e.pageContent, pageNumber: e.metadata?.pageNumber ?? e.metadata?.loc?.pageNumber })).join("\n\n")}
  `;

  const llmResponse = await openai.chat.completions.create({
    model: "gpt-6-luna",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userQuery },
    ],
  });

  console.log(`LLM Response:\n`, llmResponse.choices[0].message.content);
  return llmResponse.choices[0].message.content;
}

query("What is the main topic of the document?");
