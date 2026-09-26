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
 * Analyzes the user's natural language prompt and conversation context:
 * - Rewrites the query to be standalone.
 * - Extracts structured metadata constraints (pageNumber, year, documentType, source, author, section).
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
 * Converts a standard filter object into Chroma's 'where' query syntax.
 * Single filter: { pageNumber: 2 }
 * Multi filter:  { "$and": [{ pageNumber: 2 }, { year: 2025 }] }
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
 * Filters the document set before indexing in BM25 based on metadata constraints.
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

  // If filter matched documents, constrain to them; otherwise fallback gracefully
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
 * 6. Main Metadata-Aware Conversational RAG Pipeline
 *
 * @param {string} userQuery - Natural language query (e.g., "Find router setup from page 2")
 * @param {Object} options
 * @param {Array} [options.conversationHistory=[]] - Prior messages in the conversation
 * @param {Object} [options.explicitFilters={}] - Manual override filters (e.g. { year: 2025, documentType: "architecture" })
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

  // Merge extracted filters with any programmatic/explicit filters
  const effectiveFilters = { ...extractedFilters, ...explicitFilters };

  console.log(`Clean Search Query: "${searchQuery}"`);
  console.log(`Active Metadata Filters:`, JSON.stringify(effectiveFilters));
  console.log(`---------------------------------------------------------------`);

  // Step 2: Vector Store Setup & Filter Translation
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
  console.log(`Chroma 'where' constraint:`, chromaFilter ? JSON.stringify(chromaFilter) : "None");

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

  // Apply metadata filter to candidate documents for BM25
  const eligibleBM25Docs = filterDocuments(sanitizedDocuments, effectiveFilters);
  console.log(`Eligible documents for BM25 after metadata filter: ${eligibleBM25Docs.length}/${sanitizedDocuments.length}`);

  const bm25Retriever = BM25Retriever.fromDocuments(eligibleBM25Docs, { k: 50 });

  // Step 4: Metadata-Aware Hybrid Retrieval (Vector + BM25)
  console.log(`Executing filtered hybrid retrieval...`);
  const [vectorResults, bm25Results] = await Promise.all([
    vectorStore.similaritySearch(searchQuery, 50, chromaFilter),
    bm25Retriever.invoke(searchQuery),
  ]);

  console.log(`- Vector search returned: ${vectorResults.length} chunks (metadata constrained)`);
  console.log(`- BM25 search returned: ${bm25Results.length} chunks (metadata constrained)`);

  // Step 5: RRF Fusion
  const candidatePool = reciprocalRankFusion([vectorResults, bm25Results], 60, [0.5, 0.5]).slice(0, 50);
  console.log(`Candidate Pool size: ${candidatePool.length} chunks`);

  // Step 6: Reranker
  const topContextDocs = await rerankCandidates(searchQuery, candidatePool, 5);

  // Step 7: Final LLM Generation
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
    searchQuery,
    filters: effectiveFilters,
    sources: topContextDocs.map((d) => ({
      source: d.metadata?.source,
      pageNumber: d.metadata?.pageNumber ?? d.metadata?.loc?.pageNumber,
    })),
  };
}

// Verification Test: Query with implicit metadata constraint
async function runTest() {
  await askQuestion("Find the router connection setup details from page 2");
}

runTest();
