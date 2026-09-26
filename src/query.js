import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import OpenAI from "openai";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { fileURLToPath } from "node:url";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * ============================================================================
 * STAGE 1: CONVERSATION HISTORY -> QUERY REWRITING
 * ============================================================================
 * Analyzes conversation history and reformulates ambiguous user questions
 * into explicit, standalone search queries for dense (vector) and sparse (BM25) retrieval.
 */
async function rewriteQueryWithHistory(userQuery, conversationHistory = []) {
  console.log(`\n---------------------------------------------------------------`);
  console.log(`[STAGE 1: Query Rewriter]`);
  console.log(`- Conversation History: ${conversationHistory.length} previous messages`);
  console.log(`- Original User Query:  "${userQuery}"`);

  if (conversationHistory.length === 0) {
    console.log(`- Standalone Query:    "${userQuery}" (No history needed)`);
  }

  const prompt = `
You are an expert search query reformulation engine for a RAG retrieval system.
Given the conversation history and the latest user query:

1. "rewrittenQuery": Formulate a standalone, self-contained search query suitable for semantic vector and BM25 keyword retrieval.
   - Resolve all ambiguous pronouns ("it", "they", "its", "that device", "these steps") by replacing them with the explicit entities from earlier turns.
   - Preserve technical keywords, model names, interface names, and page references.
   - Do NOT answer the question. Only rewrite it.

2. "filters": Extract any explicit metadata constraints:
   - pageNumber: integer (e.g. 2 for "page 2")
   - year: integer
   - documentType: string
   - source: string

Return JSON ONLY in this schema:
{
  "rewrittenQuery": "rewritten standalone search terms",
  "filters": {
    "pageNumber": 2
  }
}

Conversation History:
${
  conversationHistory.length > 0
    ? conversationHistory.map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n")
    : "(No prior history - First turn)"
}

Latest User Query: "${userQuery}"
`;

  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });

    const parsed = JSON.parse(response.choices[0].message.content);
    const rewrittenQuery = parsed.rewrittenQuery || userQuery;
    const filters = parsed.filters || {};

    if (conversationHistory.length > 0) {
      console.log(`- Rewritten for Search: "${rewrittenQuery}"`);
    }
    if (Object.keys(filters).length > 0) {
      console.log(`- Extracted Filters:   ${JSON.stringify(filters)}`);
    }

    return { rewrittenQuery, filters };
  } catch (err) {
    console.warn("⚠️ Query rewriting failed, using original:", err.message);
    return { rewrittenQuery: userQuery, filters: {} };
  }
}

/**
 * ============================================================================
 * STAGE 2: METADATA-AWARE HYBRID RETRIEVAL (VECTOR + BM25)
 * ============================================================================
 */
function buildChromaFilter(filters) {
  if (!filters || Object.keys(filters).length === 0) return undefined;
  const entries = Object.entries(filters).filter(([_, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return undefined;
  if (entries.length === 1) return { [entries[0][0]]: entries[0][1] };
  return { "$and": entries.map(([k, v]) => ({ [k]: v })) };
}

function filterDocumentsForBM25(docs, filters) {
  if (!filters || Object.keys(filters).length === 0) return docs;
  const entries = Object.entries(filters).filter(([_, v]) => v !== undefined && v !== null && v !== "");
  if (entries.length === 0) return docs;

  const matched = docs.filter((doc) => {
    return entries.every(([key, value]) => {
      const docVal = doc.metadata?.[key] ?? (key === "pageNumber" ? doc.metadata?.loc?.pageNumber : undefined);
      if (docVal === undefined) return false;
      if (typeof value === "string") return String(docVal).toLowerCase().includes(value.toLowerCase());
      return docVal === value;
    });
  });
  return matched.length > 0 ? matched : docs;
}

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
 * ============================================================================
 * STAGE 3: CROSS-ENCODER RERANKER
 * ============================================================================
 */
async function rerankCandidates(query, candidates, topK = 5) {
  if (!candidates || candidates.length === 0) return [];

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

Respond ONLY with JSON:
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
      const evalResult = scoreMap.get(idx) || { score: 0, reason: "No score assigned" };
      return { doc, rerankScore: evalResult.score, reason: evalResult.reason };
    });

    scoredCandidates.sort((a, b) => b.rerankScore - a.rerankScore);

    console.log(`\n[STAGE 3: Reranker Top Results]`);
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
 * ============================================================================
 * STAGE 4: CONTEXTUAL COMPRESSION
 * ============================================================================
 */
async function compressContext(query, documents) {
  if (!documents || documents.length === 0) return [];

  const initialChars = documents.reduce((acc, d) => acc + d.pageContent.length, 0);

  const prompt = `
You are a context compression engine.
User Query: "${query}"

Extract ONLY exact facts, instructions, steps, numbers, and statements that directly answer the query.
Discard all boilerplate, extraneous descriptions, unrelated topics, headers, and filler.
If a chunk has NO relevant info, return null for it.

Output JSON ONLY:
{
  "compressedChunks": [
    { "id": 0, "compressedContent": "Extracted relevant statements only..." }
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

    console.log(`\n[STAGE 4: Contextual Compression]`);
    console.log(`- Noise reduction: ${initialChars} chars -> ${finalChars} chars (${reductionPercent}% compressed)`);
    console.log(`- Retained ${compressedDocs.length}/${documents.length} high-density chunks.`);

    return compressedDocs.length > 0 ? compressedDocs : documents;
  } catch (err) {
    console.warn("⚠️ Compression failed, falling back to raw chunks:", err.message);
    return documents;
  }
}

/**
 * ============================================================================
 * FULL CONVERSATIONAL RAG PIPELINE
 * ============================================================================
 */
export async function askQuestion(userQuery, conversationHistory = []) {
  // Step 1: Conversation History -> Query Rewriting
  const { rewrittenQuery, filters } = await rewriteQueryWithHistory(userQuery, conversationHistory);

  // Step 2: Retrieval Setup
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

  const chromaFilter = buildChromaFilter(filters);

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

  const eligibleBM25Docs = filterDocumentsForBM25(sanitizedDocuments, filters);
  const bm25Retriever = BM25Retriever.fromDocuments(eligibleBM25Docs, { k: 50 });

  // Execute Hybrid Search with the rewritten standalone query
  console.log(`\n[STAGE 2: Hybrid Retrieval]`);
  console.log(`Searching with rewritten query: "${rewrittenQuery}"...`);
  const [vectorResults, bm25Results] = await Promise.all([
    vectorStore.similaritySearch(rewrittenQuery, 50, chromaFilter),
    bm25Retriever.invoke(rewrittenQuery),
  ]);

  console.log(`- Vector retrieved: ${vectorResults.length} chunks`);
  console.log(`- BM25 retrieved:   ${bm25Results.length} chunks`);

  // Fuse candidates
  const candidatePool = reciprocalRankFusion([vectorResults, bm25Results], 60, [0.5, 0.5]).slice(0, 50);

  // Step 3: Reranker
  const topRankedDocs = await rerankCandidates(rewrittenQuery, candidatePool, 5);

  // Step 4: Contextual Compression
  const compressedContextDocs = await compressContext(rewrittenQuery, topRankedDocs);

  // Step 5: Final LLM Generation
  const SYSTEM_PROMPT = `
You are an expert AI assistant answering questions based on the provided document context and conversation history.
- Answer accurately and concisely based strictly on the context.
- Always cite the source document and page number.

Retrieved Context (Compressed to relevant facts only):
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

  console.log(`\n[FINAL ANSWER]`);
  console.log(answer);
  console.log(`---------------------------------------------------------------\n`);

  return {
    answer,
    rewrittenQuery,
    filters,
  };
}

/**
 * ============================================================================
 * INTERACTIVE CLI & MULTI-TURN DEMONSTRATION
 * ============================================================================
 */
async function runInteractiveSession() {
  const isInteractive = process.argv.includes("--interactive") || process.argv.includes("-i");

  if (isInteractive) {
    const rl = readline.createInterface({ input, output });
    const conversationHistory = [];

    console.log("\n===============================================================");
    console.log("PDF-QA Conversational RAG Session Started");
    console.log("Type your questions below. Type 'exit' or 'quit' to quit.");
    console.log("===============================================================\n");

    while (true) {
      const userQuestion = await rl.question("You: ");
      if (!userQuestion || ["exit", "quit", "q"].includes(userQuestion.trim().toLowerCase())) {
        break;
      }

      const result = await askQuestion(userQuestion, conversationHistory);
      conversationHistory.push({ role: "user", content: userQuestion });
      conversationHistory.push({ role: "assistant", content: result.answer });
    }
    rl.close();
  } else {
    // Automated 3-turn demonstration showcasing Conversation History -> Query Rewriting -> Retrieval
    console.log("\n===============================================================");
    console.log("MULTI-TURN CONVERSATIONAL RAG DEMONSTRATION");
    console.log("Pattern: Conversation History -> Query Rewriting -> Retrieval");
    console.log("===============================================================");

    const conversationHistory = [];

    // Turn 1: Initial Question
    const turn1 = "What router model is introduced on page 2?";
    const res1 = await askQuestion(turn1, conversationHistory);
    conversationHistory.push({ role: "user", content: turn1 });
    conversationHistory.push({ role: "assistant", content: res1.answer });

    // Turn 2: Follow-up with pronoun "it"
    const turn2 = "What interface does it use?";
    const res2 = await askQuestion(turn2, conversationHistory);
    conversationHistory.push({ role: "user", content: turn2 });
    conversationHistory.push({ role: "assistant", content: res2.answer });

    // Turn 3: Follow-up referring back to the interface found in turn 2
    const turn3 = "What cable connects to that interface?";
    const res3 = await askQuestion(turn3, conversationHistory);
    conversationHistory.push({ role: "user", content: turn3 });
    conversationHistory.push({ role: "assistant", content: res3.answer });
  }
}

// Run if invoked directly
const currentFilePath = fileURLToPath(import.meta.url);
if (process.argv[1] && currentFilePath.endsWith(process.argv[1].replace(/^[.\/\\]+/, ""))) {
  runInteractiveSession();
}
