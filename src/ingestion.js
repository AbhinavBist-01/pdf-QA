import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { CloudClient } from "chromadb";

async function generateEmbeddingsForFile(filePath) {
  const loader = new PDFLoader(filePath);
  const documents = await loader.load();

  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await Chroma.fromDocuments(documents, embeddings, {
    client: new CloudClient({
      apiKey: process.env.CHROMADB_API_KEY,
      tenant: "ae7af065-af71-456d-8c9c-3127e359d578",
      database: "pdf-qa",
    }),
  });
  await vectorStore.addDocuments(document);
  console.log(`All the documents are indexed....`);
}

generateEmbeddingsForFile("src/sample.pdf");
