import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";
import path from "node:path";

async function ingestFile(filePath) {
  console.log(`\n===============================================================`);
  console.log(`Starting Ingestion: "${filePath}"`);
  console.log(`===============================================================`);

  const fileName = path.basename(filePath);
  console.log(`Loading PDF from ${filePath}...`);
  const loader = new PDFLoader(filePath);
  const rawDocs = await loader.load();
  console.log(`Loaded ${rawDocs.length} raw pages.`);

  console.log("Splitting document into granular chunks (size=1000, overlap=150)...");
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 150,
  });

  const splitDocs = await splitter.splitDocuments(rawDocs);
  console.log(`Created ${splitDocs.length} chunks.`);

  const sanitizedDocuments = splitDocs.map((doc, idx) => ({
    ...doc,
    metadata: {
      source: String(doc.metadata?.source || filePath),
      fileName: fileName,
      pageNumber: Number(doc.metadata?.loc?.pageNumber || 1),
      totalPages: Number(doc.metadata?.pdf?.totalPages || rawDocs.length),
      chunkId: idx,
    },
  }));

  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  console.log("Connecting to ChromaDB Cloud collection 'pdf-qa'...");
  const vectorStore = await Chroma.fromExistingCollection(embeddings, {
    collectionName: "pdf-qa",
    index: new CloudClient({
      apiKey: process.env.CHROMADB_API_KEY,
      tenant: "ae7af065-af71-456d-8c9c-3127e359d578",
      database: "pdf-qa",
    }),
  });

  const BATCH_SIZE = 100;
  const totalBatches = Math.ceil(sanitizedDocuments.length / BATCH_SIZE);
  console.log(`Uploading ${sanitizedDocuments.length} chunks in ${totalBatches} batches (${BATCH_SIZE} per batch)...`);

  for (let i = 0; i < sanitizedDocuments.length; i += BATCH_SIZE) {
    const batch = sanitizedDocuments.slice(i, i + BATCH_SIZE);
    const batchNum = Math.floor(i / BATCH_SIZE) + 1;
    process.stdout.write(`  Uploading batch ${batchNum}/${totalBatches} (chunks ${i + 1}-${i + batch.length})... `);
    await vectorStore.addDocuments(batch);
    console.log("Done.");
  }

  console.log(`\nSuccessfully indexed all ${sanitizedDocuments.length} chunks from "${fileName}" into ChromaDB!`);
  return vectorStore;
}

// Ingest target file specified via CLI or default to 48 laws.pdf
const targetFile = process.argv[2] || "documents/48 laws.pdf";
ingestFile(targetFile).catch((err) => {
  console.error("Ingestion failed:", err);
  process.exit(1);
});
