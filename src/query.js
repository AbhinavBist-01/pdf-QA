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
 * 1. Query Analyzer & Metadata Filter Extractor (Self-Query)
 */
async function analyzeQueryAndExtractFilters(userQuery, conversationHistory = []) {
  const prompt = `
You are an intelligent query analysis agent for a RAG retrieval system.
Analyze the user's query and optional conversation history:
1. "searchQuery": Rewrite the user's query into an unambiguous, clean search phrase for keyword & semantic matching (remove phrases like "from page 2", "in 2025 docs", and resolve pronouns like "it", "they").
2. "filters": Extract any explicit metadata constraints mentioned or implied.

Supported metadata filter attributes:
- pageNumber: integer (e.g., 2 for "page 2", "second page")
- year: integer (e.g., 2025, 2024)
- documentType: string (e.g., "architecture", "lab-manual", "whitepaper", "specification")
- source: string (e.g., "cn.pdf")
- author: string
- section: string

Output JSON ONLY in this format:
{
  "searchQuery": "clean search terms",
  "filters": {
    "pageNumber": 2
  }
}
If no metadata constraints are mentioned, "filters" should be {}.

Conversation History:
${conversationHistory.map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n")}

User Query: "${userQuery}"
`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });

    const parsed = JSON.parse(response.choices[0].message.content);
    return {
      searchQuery: parsed.searchQuery || userQuery,
      filters: parsed.filters || {},
    };
  } catch (err) {
    console.warn("⚠️ Query analysis failed, proceeding without filters:", err.message);
    return { searchQuery: userQuery, filters: {} };
  }
}

/**
 * 2. ChromaDB Filter Builder
 */
function buildChromaFilter(filters) {
  if (!filters || Object.keys(filters).length === 0) {
    return undefined;
  }

  const validEntries = Object.entries(filters).filter(
    ([_, v]) => v !== undefined && v !== null && v !== ""
  );

  if (validEntries.length === 0) return undefined;

  if (validEntries.length === 1) {
    const [key, value] = validEntries[0];
    return { [key]: value };
  }

  return {
    "$and": validEntries.map(([key, value]) => ({ [key]: value })),
  };
}

/**
 * 3. BM25 Metadata Filter
 */
function filterDocuments(docs, filters) {
  if (!filters || Object.keys(filters).length === 0) {
    return docs;
  }

  const validEntries = Object.entries(filters).filter(
    ([_, v]) => v !== undefined && v !== null && v !== ""
  );

  if (validEntries.length === 0) return docs;

  const matched = docs.filter((doc) => {
    return validEntries.every(([key, value]) => {
      const docVal =
        doc.metadata?.[key] ??
        (key === "pageNumber" ? doc.metadata?.loc?.pageNumber : undefined);

      if (docVal === undefined) return false;
      if (typeof value === "string") {
        return String(docVal).toLowerCase().includes(value.toLowerCase());
      }
      return docVal === value;
    });
  });

  return matched.length > 0 ? matched : docs;
}

/**
 * 4. Reciprocal Rank Fusion (RRF)
 */
function reciprocalRankFusion(rankingsList, k = 60, weights = [0.5, 0.5]) {
  const docMap = new Map();

  rankingsList.forEach((ranking, listIdx) => {
    const weight = weights[listIdx] ?? 1.0;

    ranking.forEach((doc, rankIdx) => {
      const rank = rankIdx + 1;
      const rrfScore = weight / (rank + k);

      const source = doc.metadata?.source || "unknown";
      const page = doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? 0;
      const key = `${source}::p${page}::${doc.pageContent.trim()}`;

      if (!docMap.has(key)) {
        docMap.set(key, { document: doc, score: 0, ranks: {} });
      }

      const entry = docMap.get(key);
      entry.score += rrfScore;
      entry.ranks[`source_${listIdx}`] = rank;
    });
  });

  return Array.from(docMap.values())
    .sort((a, b) => b.score - a.score)
    .map((item) => item.document);
}

/**
 * 5. Cross-Encoder Reranker
 */
async function rerankCandidates(query, candidates, topK = 5) {
  if (!candidates || candidates.length === 0) return [];

  console.log(`Evaluating ${candidates.length} candidates in Reranker against: "${query}"...`);

  const candidateExcerpts = candidates
    .map((doc, idx) => {
      const page = doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? "?";
      const content = doc.pageContent.replace(/\s+/g, " ").trim().slice(0, 500);
      return `[Candidate ID ${idx}] (Page ${page}):\n${content}`;
    })
    .join("\n\n");

  const prompt = `
You are an expert search relevance evaluator (Cross-Encoder Reranker).
Search Query: "${query}"

Score each candidate passage's relevance to answering the search query from 0 to 100:
- 90-100: Direct, comprehensive answer.
- 60-89: Highly relevant supporting context.
- 20-59: Tangentially related topic or keyword match.
- 0-19: Irrelevant or wrong topic.

Respond ONLY with a JSON object in this exact schema:
{
  "scores": [
    { "id": 0, "score": 95, "reason": "Direct description of..." }
  ]
}

Candidates:
${candidateExcerpts}
`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });

    const parsed = JSON.parse(response.choices[0].message.content);
    const scoreMap = new Map();
    if (Array.isArray(parsed.scores)) {
      for (const item of parsed.scores) {
        scoreMap.set(item.id, { score: item.score ?? 0, reason: item.reason ?? "" });
      }
    }

    const scoredCandidates = candidates.map((doc, idx) => {
      const evaluation = scoreMap.get(idx) || { score: 0, reason: "No score assigned" };
      return {
        doc,
        rerankScore: evaluation.score,
        reason: evaluation.reason,
      };
    });

    scoredCandidates.sort((a, b) => b.rerankScore - a.rerankScore);

    console.log(`Top ${Math.min(topK, scoredCandidates.length)} after Reranking:`);
    scoredCandidates.slice(0, topK).forEach((item, i) => {
      const page = item.doc.metadata?.pageNumber ?? item.doc.metadata?.loc?.pageNumber;
      console.log(`  #${i + 1} | Score: ${item.rerankScore}/100 | Page: ${page} | Reason: ${item.reason}`);
    });

    return scoredCandidates.slice(0, topK).map((item) => item.doc);
  } catch (error) {
    console.warn("⚠️ Reranker error, using RRF order:", error.message);
    return candidates.slice(0, topK);
  }
}

/**
 * 6. Contextual Compression
 * Takes retrieved top-K chunks and extracts ONLY the facts/sentences
 * directly relevant to answering the query, removing fluff, headers, and noise.
 *
 * @param {string} query - The user search query
 * @param {Array<Document>} documents - Top K retrieved chunks
 * @returns {Promise<Array<{ pageContent: string, metadata: Object }>>} Compressed high-density context
 */
async function compressContext(query, documents) {
  if (!documents || documents.length === 0) return [];

  console.log(`\n--- Contextual Compression Stage ---`);
  console.log(`Compressing ${documents.length} chunks to isolate query-relevant information...`);

  const initialChars = documents.reduce((acc, d) => acc + d.pageContent.length, 0);

  const prompt = `
You are an expert context compression engine for a RAG retrieval system.
User Query: "${query}"

Your task:
Carefully read each document chunk.
1. Extract ONLY the exact facts, instructions, steps, numbers, and statements that directly help answer the query.
2. Discard all boilerplate, extraneous descriptions, unrelated topics, headers, and filler.
3. If an entire chunk has NO relevant information for the query, return null for that chunk.
4. Keep the extracted text factual and concise without rewording or hallucinating.

Output JSON ONLY in this format:
{
  "compressedChunks": [
    {
      "id": 0,
      "compressedContent": "Extracted relevant statements only..."
    }
  ]
}

Input Chunks:
${documents
  .map((doc, idx) => {
    const page = doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? "?";
    return `[Chunk ID ${idx}] (Page ${page}):\n${doc.pageContent}`;
  })
  .join("\n\n")}
`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });

    const parsed = JSON.parse(response.choices[0].message.content);
    const compressedMap = new Map();
    if (Array.isArray(parsed.compressedChunks)) {
      for (const item of parsed.compressedChunks) {
        if (item.compressedContent && item.compressedContent.trim().length > 0) {
          compressedMap.set(item.id, item.compressedContent.trim());
        }
      }
    }

    const compressedDocs = [];
    documents.forEach((doc, idx) => {
      if (compressedMap.has(idx)) {
        compressedDocs.push({
          pageContent: compressedMap.get(idx),
          metadata: doc.metadata,
        });
      }
    });

    const finalChars = compressedDocs.reduce((acc, d) => acc + d.pageContent.length, 0);
    const reductionPercent = initialChars > 0 ? Math.round(((initialChars - finalChars) / initialChars) * 100) : 0;

    console.log(`Compression result: ${initialChars} chars -> ${finalChars} chars (${reductionPercent}% noise removed)`);
    console.log(`Retained ${compressedDocs.length}/${documents.length} high-density chunks.`);

    return compressedDocs.length > 0 ? compressedDocs : documents;
  } catch (err) {
    console.warn("⚠️ Context compression failed, falling back to raw chunks:", err.message);
    return documents;
  }
}

/**
 * 7. Main Conversational RAG Query Pipeline with Contextual Compression
 */
export async function askQuestion(userQuery, options = {}) {
  const { conversationHistory = [], explicitFilters = {} } = options;

  console.log(`\n===============================================================`);
  console.log(`User Query: "${userQuery}"`);

  // Step 1: Query Analysis & Metadata Filter Extraction
  const { searchQuery, filters: extractedFilters } = await analyzeQueryAndExtractFilters(
    userQuery,
    conversationHistory
  );

  const effectiveFilters = { ...extractedFilters, ...explicitFilters };

  console.log(`Clean Search Query: "${searchQuery}"`);
  console.log(`Active Metadata Filters:`, JSON.stringify(effectiveFilters));
  console.log(`---------------------------------------------------------------`);

  // Step 2: Chroma Vector Store Setup
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

  const chromaFilter = buildChromaFilter(effectiveFilters);

  // Step 3: Document Loading & Metadata-Filtered BM25 Setup
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

  const eligibleBM25Docs = filterDocuments(sanitizedDocuments, effectiveFilters);
  const bm25Retriever = BM25Retriever.fromDocuments(eligibleBM25Docs, { k: 50 });

  // Step 4: Parallel Filtered Hybrid Retrieval
  console.log(`Executing filtered hybrid retrieval...`);
  const [vectorResults, bm25Results] = await Promise.all([
    vectorStore.similaritySearch(searchQuery, 50, chromaFilter),
    bm25Retriever.invoke(searchQuery),
  ]);

  console.log(`- Vector search returned: ${vectorResults.length} chunks`);
  console.log(`- BM25 search returned: ${bm25Results.length} chunks`);

  // Step 5: Reciprocal Rank Fusion (RRF)
  const candidatePool = reciprocalRankFusion([vectorResults, bm25Results], 60, [0.5, 0.5]).slice(0, 50);
  console.log(`Candidate Pool size: ${candidatePool.length} chunks`);

  // Step 6: Reranker Stage -> Top 5 Chunks
  const topRankedDocs = await rerankCandidates(searchQuery, candidatePool, 5);

  // Step 7: Contextual Compression -> Keep only query-relevant facts
  const compressedContextDocs = await compressContext(searchQuery, topRankedDocs);

  // Step 8: Final LLM Generation
  const SYSTEM_PROMPT = `
You are an expert assistant answering questions based on the provided document context and conversation history.
Do not hallucinate or answer outside the provided context.
Always state the source document and page number where you found the answer.

Context Documents (Compressed to relevant facts only):
${compressedContextDocs
  .map((doc) =>
    JSON.stringify({
      bookName: doc.metadata?.source,
      pageContent: doc.pageContent,
      pageNumber: doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber,
    })
  )
  .join("\n\n")}
`;

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...conversationHistory,
    { role: "user", content: userQuery },
  ];

  const llmResponse = await openai.chat.completions.create({
    model: "gpt-6-luna",
    messages: messages,
  });

  const answer = llmResponse.choices[0].message.content;

  console.log(`\n===============================================================`);
  console.log(`Final LLM Response:\n${answer}`);
  console.log(`===============================================================\n`);

  return {
    answer,
    searchQuery,
    filters: effectiveFilters,
    sources: compressedContextDocs.map((d) => ({
      source: d.metadata?.source,
      pageNumber: d.metadata?.pageNumber ?? d.metadata?.loc?.pageNumber,
    })),
  };
}

// Verification Test: Query with broad retrieved text compressed to exact answer
async function runTest() {
  await askQuestion("What cable type and router interface are used to connect the PC?");
}

runTest();
