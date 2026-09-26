import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Reciprocal Rank Fusion (RRF) algorithm
 * Combines multiple ranked lists into a single ranked list.
 *
 * Formula:
 * RRF_score(d) = Σ [ weight_i / (rank_i(d) + k) ]
 *
 * @param {Array<Array<any>>} rankingsList - Array of ranked document arrays [[doc1, doc2, ...], [docA, docB, ...]]
 * @param {number} k - Constant smoothing parameter (standard default: 60) to prevent top items from dominating
 * @param {Array<number>} weights - Relative weights for each ranking source (e.g., [0.5, 0.5])
 * @returns {Array<{ document: any, score: number, ranks: Object }>} Fused and sorted results
 */
function reciprocalRankFusion(rankingsList, k = 60, weights = [0.5, 0.5]) {
  const docMap = new Map(); // key -> { document, score, ranks }

  rankingsList.forEach((ranking, listIdx) => {
    const weight = weights[listIdx] ?? 1.0;

    ranking.forEach((doc, rankIdx) => {
      const rank = rankIdx + 1; // 1-based rank (1st, 2nd, 3rd, ...)
      const rrfScore = weight / (rank + k);

      // Create a unique key for deduplication based on content and source metadata
      const source = doc.metadata?.source || "unknown";
      const page = doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? 0;
      const key = `${source}::p${page}::${doc.pageContent.trim()}`;

      if (!docMap.has(key)) {
        docMap.set(key, {
          document: doc,
          score: 0,
          ranks: {},
        });
      }

      const entry = docMap.get(key);
      entry.score += rrfScore;
      entry.ranks[`source_${listIdx}`] = rank;
    });
  });

  // Sort documents descending by total RRF score
  return Array.from(docMap.values()).sort((a, b) => b.score - a.score);
}

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

  const bm25Retriever = BM25Retriever.fromDocuments(sanitizedDocuments, { k: 5 });

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
    60,         // smoothing constant (k)
    [0.5, 0.5]  // 50% semantic, 50% keyword weight
  );

  console.log("\n--- Top Fused Chunks (RRF Scoring) ---");
  fusedItems.slice(0, 5).forEach((item, idx) => {
    const page = item.document.metadata?.pageNumber ?? item.document.metadata?.loc?.pageNumber;
    console.log(
      `#${idx + 1} | Score: ${item.score.toFixed(5)} | Page: ${page} | Ranks: Vector=#${item.ranks.source_0 ?? "N/A"}, BM25=#${item.ranks.source_1 ?? "N/A"}`
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
        })
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

  console.log(`\n--- LLM Response ---\n`, llmResponse.choices[0].message.content);
  return llmResponse.choices[0].message.content;
}

query("What is the main topic of the document page 2?");
