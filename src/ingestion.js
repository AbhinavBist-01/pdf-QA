import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";

async function generateEmbeddingsForFile(filePath) {
  console.log(`Loading document from: ${filePath}...`);
  const loader = new PDFLoader(filePath);
  const documents = await loader.load();
  console.log(`Loaded ${documents.length} pages/chunks.`);

  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const sanitizedDocuments = documents.map((doc) => ({
    ...doc,
    metadata: {
      source: String(doc.metadata?.source || filePath),
      pageNumber: Number(doc.metadata?.loc?.pageNumber || 1),
      totalPages: Number(doc.metadata?.pdf?.totalPages || 1),
    },
  }));

  const vectorStore = await Chroma.fromDocuments(sanitizedDocuments, embeddings, {
    collectionName: "pdf-qa",
    index: new CloudClient({
      apiKey: process.env.CHROMADB_API_KEY,
      tenant: "ae7af065-af71-456d-8c9c-3127e359d578",
      database: "pdf-qa",
    }),
  });

  console.log(`All the documents are indexed into Chroma collection 'pdf-qa'.`);
  return vectorStore;
}

generateEmbeddingsForFile("documents/cn.pdf");
