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
 * 1. Query Rewriter
 * Takes conversation history and the latest user query, then rewrites it
 * into an unambiguous, self-contained search query.
 *
 * Resolves anaphoras and pronouns ("it", "they", "that", "its", "the device")
 * using prior turns so hybrid search and BM25 can match keywords effectively.
 */
async function rewriteQuery(conversationHistory, latestQuery) {
  if (!conversationHistory || conversationHistory.length === 0) {
    return latestQuery;
  }

  const prompt = `
You are a search query reformulation expert for a RAG retrieval system.
Given the conversation history and a follow-up user query, rewrite the follow-up query into an independent, self-contained search query.

Rules:
1. Resolve all pronouns and ambiguous references ("it", "they", "its", "this setup", "that router") using details from the conversation.
2. Include specific entity names, page numbers, technical terms, and acronyms mentioned earlier.
3. Do NOT answer the question.
4. Output ONLY the rewritten search query with no quotes, preamble, or commentary.
5. If the query is already clear and self-contained, return it unchanged.

Conversation History:
${conversationHistory.map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n")}

User Follow-up Query: "${latestQuery}"
Rewritten Standalone Query:`;

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    temperature: 0,
    messages: [{ role: "user", content: prompt }],
  });

  const rewritten = response.choices[0].message.content.trim().replace(/^["']|["']$/g, "");
  return rewritten;
}

/**
 * 2. Reciprocal Rank Fusion (RRF)
 * Merges multiple ranked lists into a unified list.
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
 * 3. Cross-Encoder Reranker
 * Evaluates candidate chunks against the search query, assigning 0-100 relevance scores.
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
 * 4. Main Conversational RAG Query Pipeline
 */
export async function askQuestion(userQuery, conversationHistory = []) {
  console.log(`\n===============================================================`);
  console.log(`Original User Query: "${userQuery}"`);
  console.log(`Conversation History Length: ${conversationHistory.length} turns`);

  // Step 1: Query Rewriter
  const searchReadyQuery = await rewriteQuery(conversationHistory, userQuery);
  console.log(`Rewritten Search Query: "${searchReadyQuery}"`);
  console.log(`---------------------------------------------------------------`);

  // Step 2: Setup Hybrid Retrievers
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

  const vectorStoreRetriever = vectorStore.asRetriever({ k: 50 });

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

  const bm25Retriever = BM25Retriever.fromDocuments(sanitizedDocuments, { k: 50 });

  // Step 3: Hybrid Search (Vector + BM25) using rewritten query
  console.log(`Executing Hybrid Search with: "${searchReadyQuery}"...`);
  const [vectorResults, bm25Results] = await Promise.all([
    vectorStoreRetriever.invoke(searchReadyQuery),
    bm25Retriever.invoke(searchReadyQuery),
  ]);

  // Step 4: RRF Fusion -> Candidate pool
  const candidatePool = reciprocalRankFusion([vectorResults, bm25Results], 60, [0.5, 0.5]).slice(0, 50);
  console.log(`Candidate Pool size: ${candidatePool.length} chunks`);

  // Step 5: Reranker -> Top 5
  const topContextDocs = await rerankCandidates(searchReadyQuery, candidatePool, 5);

  // Step 6: Final LLM Generation (incorporating conversation history + retrieved context)
  const SYSTEM_PROMPT = `
You are an expert assistant answering questions based on the provided document context and conversation history.
Do not hallucinate or answer outside the provided context.
Always state the source document and page number where you found the answer.

Context Documents:
${topContextDocs
  .map((doc) =>
    JSON.stringify({
      bookName: doc.metadata?.source,
      pageContent: doc.pageContent,
      pageNumber: doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber,
    })
  )
  .join("\n\n")}
`;

  // Build message history for the final LLM response
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

  console.log(`\nFinal LLM Response:\n${answer}`);
  console.log(`===============================================================\n`);

  return {
    answer,
    searchQuery: searchReadyQuery,
    sources: topContextDocs.map((d) => ({
      source: d.metadata?.source,
      pageNumber: d.metadata?.pageNumber ?? d.metadata?.loc?.pageNumber,
    })),
  };
}

// Multi-turn Conversation Demo
async function runDemo() {
  const conversationHistory = [];

  // Turn 1
  const query1 = "What devices are being set up on page 2?";
  const res1 = await askQuestion(query1, conversationHistory);
  conversationHistory.push({ role: "user", content: query1 });
  conversationHistory.push({ role: "assistant", content: res1.answer });

  // Turn 2: Follow-up question with ambiguous pronoun "they"
  const query2 = "How are they connected with each other?";
  const res2 = await askQuestion(query2, conversationHistory);
  conversationHistory.push({ role: "user", content: query2 });
  conversationHistory.push({ role: "assistant", content: res2.answer });
}

runDemo();
