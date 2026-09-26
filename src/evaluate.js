import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { CloudClient } from "chromadb";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import OpenAI from "openai";
import { performance } from "node:perf_hooks";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * ============================================================================
 * 1. EVALUATION BENCHMARK DATASET
 * ============================================================================
 */
export const EVAL_DATASET = [
  {
    id: "q1_hardware_setup",
    question: "What cable type and router interface are used to connect the PC in step 2?",
    conversationHistory: [],
    relevantPages: [2],
    expectedAnswer:
      "A Straight-Through cable connects the PC's FastEthernet interface to the router's GigabitEthernet0/0/0 interface.",
  },
  {
    id: "q2_definition_aim",
    question: "What does TELNET stand for and what is the aim of Experiment 5?",
    conversationHistory: [],
    relevantPages: [1],
    expectedAnswer:
      "TELNET stands for Teletype Network. The aim of Experiment 5 is to understand the operation of TELNET by accessing the router in a server room from a PC in IT Office.",
  },
  {
    id: "q3_ip_configuration",
    question: "What static IP address and default gateway are configured for the PC in step 3?",
    conversationHistory: [],
    relevantPages: [3],
    expectedAnswer:
      "The PC is configured with static IP 192.168.1.1 and default gateway 192.168.1.2.",
  },
  {
    id: "q4_conversational_pronoun",
    question: "What hostname change was verified in its final result?",
    conversationHistory: [
      { role: "user", content: "What is Experiment 5 about?" },
      { role: "assistant", content: "Experiment 5 is about accessing a router via TELNET to configure it." },
    ],
    relevantPages: [5],
    expectedAnswer:
      "The hostname was changed from cnlab to cnlab2 using TELNET.",
  },
  {
    id: "q5_metadata_filter",
    question: "List the verification step described on page 4.",
    conversationHistory: [],
    relevantPages: [4],
    expectedAnswer:
      "Ping to verify the connection after entering the bold marked CLI commands.",
  },
];

/**
 * ============================================================================
 * 2. SHARED PIPELINE UTILITIES
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
        docMap.set(key, { document: doc, score: 0 });
      }
      docMap.get(key).score += rrfScore;
    });
  });

  return Array.from(docMap.values())
    .sort((a, b) => b.score - a.score)
    .map((item) => item.document);
}

async function rewriteQuery(query, history = [], enableFilterExtraction = false) {
  const prompt = `
You are a search query reformulation expert for a RAG system.
Given conversation history and the latest user query:
1. "rewrittenQuery": Output a standalone search query. Resolve all ambiguous pronouns ("it", "they", "its", "that").
2. "filters": ${enableFilterExtraction ? 'Extract explicit metadata filters like pageNumber (integer).' : 'Return {}'}.

Output JSON ONLY:
{
  "rewrittenQuery": "...",
  "filters": {}
}

History:
${history.map((m) => `${m.role.toUpperCase()}: ${m.content}`).join("\n")}

Query: "${query}"
`;

  try {
    const res = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    const parsed = JSON.parse(res.choices[0].message.content);
    return {
      rewrittenQuery: parsed.rewrittenQuery || query,
      filters: enableFilterExtraction ? parsed.filters || {} : {},
    };
  } catch {
    return { rewrittenQuery: query, filters: {} };
  }
}

async function rerank(query, candidates, topK = 5) {
  if (!candidates || candidates.length === 0) return [];
  const candidateExcerpts = candidates
    .map((doc, idx) => {
      const page = doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? "?";
      return `[ID ${idx}] (Page ${page}):\n${doc.pageContent.slice(0, 400)}`;
    })
    .join("\n\n");

  const prompt = `
Score candidate relevance to query: "${query}" from 0 to 100.
Return JSON ONLY: {"scores": [{"id": 0, "score": 95}]}

Candidates:
${candidateExcerpts}
`;

  try {
    const res = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    const parsed = JSON.parse(res.choices[0].message.content);
    const scoreMap = new Map((parsed.scores || []).map((s) => [s.id, s.score]));
    return candidates
      .map((doc, idx) => ({ doc, score: scoreMap.get(idx) ?? 0 }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => item.doc);
  } catch {
    return candidates.slice(0, topK);
  }
}

async function compress(query, documents) {
  if (!documents || documents.length === 0) return [];
  const prompt = `
User Query: "${query}"
Extract ONLY exact facts directly answering the query from each chunk. Discard fluff. If none, return null.
Return JSON: {"compressedChunks": [{"id": 0, "compressedContent": "..."}]}

Chunks:
${documents.map((d, i) => `[ID ${i}]:\n${d.pageContent}`).join("\n\n")}
`;

  try {
    const res = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    const parsed = JSON.parse(res.choices[0].message.content);
    const compressedMap = new Map(
      (parsed.compressedChunks || [])
        .filter((c) => c.compressedContent)
        .map((c) => [c.id, c.compressedContent.trim()])
    );

    const out = [];
    documents.forEach((doc, idx) => {
      if (compressedMap.has(idx)) {
        out.push({ pageContent: compressedMap.get(idx), metadata: doc.metadata });
      }
    });
    return out.length > 0 ? out : documents;
  } catch {
    return documents;
  }
}

/**
 * ============================================================================
 * 3. EVALUATION METRICS EVALUATOR (LLM-as-a-Judge & Citation/Ranking Verification)
 * ============================================================================
 */
async function evaluateAnswerCorrectness(question, generatedAnswer, expectedAnswer) {
  const prompt = `
You are an expert judge evaluating RAG answer accuracy.
Question: "${question}"
Ground Truth Expected Answer: "${expectedAnswer}"
Generated Answer: "${generatedAnswer}"

Score the answer correctness from 0 to 100 based on factual alignment with the ground truth.
Return JSON ONLY:
{"score": 95, "reason": "Explanation"}
`;

  try {
    const res = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    const parsed = JSON.parse(res.choices[0].message.content);
    return parsed.score ?? 50;
  } catch {
    return 50;
  }
}

function evaluateCitationCorrectness(generatedAnswer, relevantPages) {
  const pageMatches = [...generatedAnswer.matchAll(/page\s*(\d+)/gi)].map((m) => Number(m[1]));
  if (pageMatches.length === 0) return 0;
  const validCitations = pageMatches.filter((p) => relevantPages.includes(p));
  return validCitations.length > 0 ? 1 : 0;
}

function computeRankingMetrics(retrievedDocs, relevantPages) {
  const retrievedPageNumbers = retrievedDocs.map(
    (d) => d.metadata?.pageNumber ?? d.metadata?.loc?.pageNumber
  );

  // Recall@K: Did we retrieve ANY of the relevant pages in top K?
  const hasRelevant = retrievedPageNumbers.some((p) => relevantPages.includes(p));
  const recallAtK = hasRelevant ? 1 : 0;

  // MRR: 1 / rank of first relevant page
  let mrr = 0;
  for (let i = 0; i < retrievedPageNumbers.length; i++) {
    if (relevantPages.includes(retrievedPageNumbers[i])) {
      mrr = 1 / (i + 1);
      break;
    }
  }

  return { recallAtK, mrr, retrievedPageNumbers };
}

/**
 * ============================================================================
 * 4. CONFIGURABLE PIPELINE RUNNER (ABLATION SUITE)
 * ============================================================================
 */
async function runRAGPipeline(testItem, config, resources) {
  const { vectorStore, allDocs } = resources;
  const startTime = performance.now();

  let activeQuery = testItem.question;
  let activeFilters = {};

  // 1. Query Rewrite / Memory
  if (config.useQueryRewrite) {
    const rewriteRes = await rewriteQuery(
      testItem.question,
      testItem.conversationHistory,
      config.useFilters
    );
    activeQuery = rewriteRes.rewrittenQuery;
    activeFilters = rewriteRes.filters;
  }

  const chromaFilter = config.useFilters ? buildChromaFilter(activeFilters) : undefined;

  let retrievedDocs = [];

  // 2. Retrieval
  if (config.useHybrid) {
    const eligibleBM25Docs = config.useFilters ? filterDocumentsForBM25(allDocs, activeFilters) : allDocs;
    const bm25Retriever = BM25Retriever.fromDocuments(eligibleBM25Docs, { k: config.retrieveK || 50 });
    const [vecResults, bm25Results] = await Promise.all([
      vectorStore.similaritySearch(activeQuery, config.retrieveK || 50, chromaFilter),
      bm25Retriever.invoke(activeQuery),
    ]);
    retrievedDocs = reciprocalRankFusion([vecResults, bm25Results], 60, [0.5, 0.5]);
  } else {
    // Baseline: pure vector search
    retrievedDocs = await vectorStore.similaritySearch(activeQuery, 5, chromaFilter);
  }

  // Measure ranking quality before post-processing
  const { recallAtK, mrr, retrievedPageNumbers } = computeRankingMetrics(retrievedDocs.slice(0, 5), testItem.relevantPages);

  // 3. Reranker
  let candidateDocs = retrievedDocs.slice(0, 5);
  if (config.useReranker) {
    candidateDocs = await rerank(activeQuery, retrievedDocs.slice(0, 50), 5);
  }

  // 4. Contextual Compression
  let contextDocs = candidateDocs;
  if (config.useCompression) {
    contextDocs = await compress(activeQuery, candidateDocs);
  }

  // Measure context character footprint (token proxy)
  const contextChars = contextDocs.reduce((acc, d) => acc + d.pageContent.length, 0);

  // 5. LLM Answer Generation
  const prompt = `
Answer the query based ONLY on the context below. State the page number citation.
Context:
${contextDocs.map((d) => `Page ${d.metadata?.pageNumber ?? d.metadata?.loc?.pageNumber}: ${d.pageContent}`).join("\n\n")}

Question: ${testItem.question}
`;

  const messages = [
    { role: "system", content: "You are a factual assistant. Cite page numbers." },
    ...(config.useQueryRewrite ? testItem.conversationHistory : []),
    { role: "user", content: prompt },
  ];

  const llmRes = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: messages,
    temperature: 0,
  });

  const generatedAnswer = llmRes.choices[0].message.content;
  const latency = Math.round(performance.now() - startTime);

  // Evaluate generated answer
  const answerCorrectness = await evaluateAnswerCorrectness(
    testItem.question,
    generatedAnswer,
    testItem.expectedAnswer
  );
  const citationCorrectness = evaluateCitationCorrectness(generatedAnswer, testItem.relevantPages);

  return {
    recallAtK,
    mrr,
    answerCorrectness,
    citationCorrectness,
    latency,
    contextChars,
    generatedAnswer,
  };
}

/**
 * ============================================================================
 * 5. BENCHMARK SUITE EXECUTION & REPORTING
 * ============================================================================
 */
async function main() {
  console.log("======================================================================");
  console.log("            RAG EVALUATION & ABLATION BENCHMARK HARNESS               ");
  console.log("======================================================================\n");
  console.log(`Test Dataset: ${EVAL_DATASET.length} representative test cases`);

  // Initialize resources
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

  const pdfLoader = new PDFLoader("documents/cn.pdf");
  const rawDocs = await pdfLoader.load();
  const allDocs = rawDocs.map((doc) => ({
    ...doc,
    metadata: {
      source: "documents/cn.pdf",
      pageNumber: Number(doc.metadata?.loc?.pageNumber || 1),
      totalPages: Number(doc.metadata?.pdf?.totalPages || 1),
    },
  }));

  const resources = { vectorStore, allDocs };

  // Define ablation configurations
  const ablationConfigs = [
    {
      name: "1. Baseline (Vector Only)",
      useHybrid: false,
      useReranker: false,
      useQueryRewrite: false,
      useFilters: false,
      useCompression: false,
    },
    {
      name: "2. + Hybrid (Vector+BM25)",
      useHybrid: true,
      useReranker: false,
      useQueryRewrite: false,
      useFilters: false,
      useCompression: false,
    },
    {
      name: "3. + Reranker",
      useHybrid: true,
      useReranker: true,
      useQueryRewrite: false,
      useFilters: false,
      useCompression: false,
    },
    {
      name: "4. + Query Rewrite / Memory",
      useHybrid: true,
      useReranker: true,
      useQueryRewrite: true,
      useFilters: false,
      useCompression: false,
    },
    {
      name: "5. + Metadata Filters",
      useHybrid: true,
      useReranker: true,
      useQueryRewrite: true,
      useFilters: true,
      useCompression: false,
    },
    {
      name: "6. Final RAG (+ Compression)",
      useHybrid: true,
      useReranker: true,
      useQueryRewrite: true,
      useFilters: true,
      useCompression: true,
    },
  ];

  const summaryResults = [];

  for (const config of ablationConfigs) {
    console.log(`\nEvaluating Configuration: [${config.name}]...`);
    const metrics = {
      recalls: [],
      mrrs: [],
      correctnessScores: [],
      citationScores: [],
      latencies: [],
      contextSizes: [],
    };

    for (const testItem of EVAL_DATASET) {
      process.stdout.write(`  - Running query: "${testItem.id}"... `);
      const res = await runRAGPipeline(testItem, config, resources);
      metrics.recalls.push(res.recallAtK);
      metrics.mrrs.push(res.mrr);
      metrics.correctnessScores.push(res.answerCorrectness);
      metrics.citationScores.push(res.citationCorrectness);
      metrics.latencies.push(res.latency);
      metrics.contextSizes.push(res.contextChars);
      console.log(`OK (${res.latency}ms, recall=${res.recallAtK}, correct=${res.answerCorrectness})`);
    }

    const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

    const row = {
      Configuration: config.name,
      "Recall@5": `${Math.round(avg(metrics.recalls) * 100)}%`,
      MRR: avg(metrics.mrrs).toFixed(3),
      "Answer Correctness": `${Math.round(avg(metrics.correctnessScores))}%`,
      "Citation Accuracy": `${Math.round(avg(metrics.citationScores) * 100)}%`,
      "Avg Latency": `${Math.round(avg(metrics.latencies))}ms`,
      "Context Chars": Math.round(avg(metrics.contextSizes)),
    };

    summaryResults.push(row);
  }

  console.log("\n======================================================================");
  console.log("                     FINAL ABLATION RESULTS TABLE                     ");
  console.log("======================================================================\n");
  console.table(summaryResults);
}

main().catch((err) => {
  console.error("Evaluation error:", err);
  process.exit(1);
});
