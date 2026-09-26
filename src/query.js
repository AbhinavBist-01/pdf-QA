import { OpenAIEmbeddings } from "@langchain/openai";
import {} from "@langchain/chromadb";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function query(userQuery) {
  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await Chroma.fromExistingCollection(
    "pdf-qa",
    embeddings,
    {
      client: new CloudClient({
        apiKey: process.env.CHROMADB_API_KEY,
        tenant: "ae7af065-af71-456d-8c9c-3127e359d578",
        database: "pdf-qa",
      }),
    },
  );
  const vectorStoreRetriever = vectorStore.asRetriever({ k: 5 });

  const results = await vectorStoreRetreiver.getRelevantDocuments(userQuery);

  const SYSTEM_PROMPT = `
    You are an expert in answereing user query based on the provided context about document.
    Do not answere anything beyond what is not provided.

    Always also answer the user in short and tell on which page number that content is available and also name of the book

    User Documents:
    ${results.map((e) => JSON.stringify({ bookName: e.metadata.source, pageContent: e.pageContent, pageNumber: e.metadata.loc.pageNumber })).join("\n\n")}
  `;

  const llmResponse = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userQuery },
    ],
  });

  console.log(`LLM Response:`, llmResponse.choices[0].message.content);
}
