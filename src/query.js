import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { reciprocalRankFusion } from "./rrf.js";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function query(userQuery) {
  console.log(`\n==================================================`);
  console.log(`User Query: "${userQuery}"`);
  console.log(`==================================================\n`);

  // 1. Vector Search (Semantic) Retriever setup via ChromaDB
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

  // 2. Keyword Search (BM25) Retriever setup
  const pdfPath = "documents/cn.pdf";
  const loader = new PDFLoader(pdfPath);
  const rawDocs = await loader.load();
  const sanitizedDocuments = rawDocs.map((doc) => ({
    ...doc,
    metadata: {
      source: String(doc.metadata?.source || pdfPath),
      pageNumber: Number(doc.metadata?.loc?.pageNumber || 1),
      totalPages: Number(doc.metadata?.pdf?.totalPages || 1),
    },
  }));

  const bm25Retriever = BM25Retriever.fromDocuments(sanitizedDocuments, {
    k: 5,
  });

  // 3. Run both retrievers in parallel
  console.log("Executing Vector Search & BM25 Keyword Search in parallel...");
  const [vectorResults, bm25Results] = await Promise.all([
    vectorStoreRetriever.invoke(userQuery),
    bm25Retriever.invoke(userQuery),
  ]);

  console.log(`\n- Vector search retrieved: ${vectorResults.length} chunks`);
  console.log(`- BM25 keyword search retrieved: ${bm25Results.length} chunks`);

  // 4. Reciprocal Rank Fusion (RRF)
  const fusedItems = reciprocalRankFusion(
    [vectorResults, bm25Results],
    60, // smoothing constant (k)
    [0.5, 0.5], // 50% semantic, 50% keyword weight
  );

  console.log("\n--- Top Fused Chunks (RRF Scoring) ---");
  fusedItems.slice(0, 5).forEach((item, idx) => {
    const page =
      item.document.metadata?.pageNumber ??
      item.document.metadata?.loc?.pageNumber;
    console.log(
      `#${idx + 1} | Score: ${item.score.toFixed(5)} | Page: ${page} | Ranks: Vector=#${item.ranks.source_0 ?? "N/A"}, BM25=#${item.ranks.source_1 ?? "N/A"}`,
    );
  });

  // 5. Select Top-K
  const topKDocs = fusedItems.slice(0, 5).map((item) => item.document);

  // 6. Format Context for LLM
  const SYSTEM_PROMPT = `
    You are an expert in answering user query based on the provided context about document.
    Do not answer anything beyond what is provided.

    Always also answer the user concisely and state which page number that content is available on and the name/source of the document.

    User Documents:
    ${topKDocs
      .map((e) =>
        JSON.stringify({
          bookName: e.metadata?.source,
          pageContent: e.pageContent,
          pageNumber: e.metadata?.pageNumber ?? e.metadata?.loc?.pageNumber,
        }),
      )
      .join("\n\n")}
  `;

  // 7. Generate Answer with LLM
  const llmResponse = await openai.chat.completions.create({
    model: "gpt-6-luna",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userQuery },
    ],
  });

  console.log(
    `\n--- LLM Response ---\n`,
    llmResponse.choices[0].message.content,
  );
  return llmResponse.choices[0].message.content;
}

query("What is the main topic of the document page 2?");
