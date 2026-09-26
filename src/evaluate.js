import "dotenv/config";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import OpenAI from "openai";
import { performance } from "node:perf_hooks";
import { readFile } from "node:fs/promises";
import { EVAL_DATASET } from "./dataset.js";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * ============================================================================
 * 1. RAG COMPONENTS & UTILITIES
 * ============================================================================
 */

function reciprocalRankFusion(rankingsList, k = 60, weights = [0.5, 0.5]) {
  const docMap = new Map();

  rankingsList.forEach((ranking, listIdx) => {
    const weight = weights[listIdx] ?? 1.0;
    ranking.forEach((doc, rankIdx) => {
      const rank = rankIdx + 1;
      const rrfScore = weight / (rank + k);
      const key = `${doc.metadata?.fileName || doc.metadata?.source}::p${doc.metadata?.pageNumber}::${doc.pageContent.slice(0, 80).trim()}`;

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
You are an expert search query reformulation assistant.
Given conversation history and the latest user query:
1. "rewrittenQuery": Output a standalone search query. Replace all ambiguous pronouns ("it", "they", "its", "that", "these", "that port", "that party", "him", "the king") with explicit entities from history.
2. "filters": ${enableFilterExtraction ? 'Extract explicit metadata filters like pageNumber (integer) or fileName ("48 laws.pdf" or "cn.pdf").' : 'Return {}'}.

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

async function rerank(query, candidates, topK = 3) {
  if (!candidates || candidates.length === 0) return [];
  const candidateExcerpts = candidates
    .map((doc, idx) => {
      const file = doc.metadata?.fileName || doc.metadata?.source || "?";
      const page = doc.metadata?.pageNumber ?? "?";
      return `[ID ${idx}] (${file}, Page ${page}):\n${doc.pageContent.slice(0, 300)}`;
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
Extract ONLY exact factual statements, parameters, or instructions directly answering the query from each chunk. Discard fluff. If a chunk contains no answer, return null for it.
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
 * 2. METRICS EVALUATORS
 * ============================================================================
 */
async function evaluateAnswerCorrectness(question, generatedAnswer, expectedAnswer) {
  const prompt = `
You are an expert judge evaluating RAG answer accuracy against ground truth.
Question: "${question}"
Ground Truth: "${expectedAnswer}"
Generated Answer: "${generatedAnswer}"

Score factual correctness from 0 to 100 based on alignment with ground truth.
Return JSON ONLY: {"score": 95}
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
  const pageMatches = [...generatedAnswer.matchAll(/page[s]?\s*(\d+)/gi)].map((m) => Number(m[1]));
  if (pageMatches.length === 0) return 0;
  return pageMatches.some((p) => relevantPages.includes(p)) ? 1 : 0;
}

function computeRankingMetrics(retrievedDocs, targetFile, relevantPages) {
  let hasRelevant = false;
  let mrr = 0;

  for (let i = 0; i < retrievedDocs.length; i++) {
    const d = retrievedDocs[i];
    const docFile = d.metadata?.fileName || d.metadata?.source || "";
    const docPage = d.metadata?.pageNumber;

    const fileMatches = !targetFile || docFile.toLowerCase().includes(targetFile.toLowerCase());
    const pageMatches = !relevantPages || relevantPages.length === 0 || relevantPages.includes(docPage);

    if (fileMatches && pageMatches) {
      if (!hasRelevant) {
        hasRelevant = true;
        mrr = 1 / (i + 1);
      }
    }
  }

  return { recall: hasRelevant ? 1 : 0, mrr };
}

/**
 * ============================================================================
 * 3. PIPELINE RUNNER
 * ============================================================================
 */
async function runRAGPipeline(testItem, config, resources) {
  const { vectorStore, chunks } = resources;
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

  // Filter function for metadata
  const matchesFilter = (doc) => {
    if (!config.useFilters) return true;
    if (activeFilters.pageNumber !== undefined && doc.metadata?.pageNumber !== activeFilters.pageNumber) {
      return false;
    }
    if (activeFilters.fileName && !doc.metadata?.fileName?.toLowerCase().includes(activeFilters.fileName.toLowerCase())) {
      return false;
    }
    return true;
  };

  let retrievedDocs = [];

  // 2. Retrieval
  if (config.useHybrid) {
    const eligibleChunks = chunks.filter(matchesFilter);
    const targetPool = eligibleChunks.length > 0 ? eligibleChunks : chunks;
    const bm25 = BM25Retriever.fromDocuments(targetPool, { k: 25 });

    const [vecResults, bm25Results] = await Promise.all([
      vectorStore.similaritySearch(activeQuery, 25),
      bm25.invoke(activeQuery),
    ]);

    retrievedDocs = reciprocalRankFusion([vecResults, bm25Results], 60, [0.5, 0.5]);
  } else {
    // Baseline: Dense Vector Search only (k=3)
    retrievedDocs = await vectorStore.similaritySearch(activeQuery, 3);
  }

  // Measure ranking metrics at top 3
  const { recall, mrr } = computeRankingMetrics(retrievedDocs.slice(0, 3), testItem.targetFile, testItem.relevantPages);

  // 3. Reranker
  let candidateDocs = retrievedDocs.slice(0, 3);
  if (config.useReranker) {
    candidateDocs = await rerank(activeQuery, retrievedDocs.slice(0, 25), 3);
  }

  // 4. Contextual Compression
  let contextDocs = candidateDocs;
  if (config.useCompression) {
    contextDocs = await compress(activeQuery, candidateDocs);
  }

  const contextChars = contextDocs.reduce((acc, d) => acc + d.pageContent.length, 0);

  // 5. LLM Answer Generation
  const prompt = `
Answer based ONLY on the context below. Always cite the document and page number.
Context:
${contextDocs.map((d) => `[${d.metadata?.fileName || d.metadata?.source}, Page ${d.metadata?.pageNumber}]: ${d.pageContent}`).join("\n\n")}

Question: ${testItem.question}
`;

  const messages = [
    { role: "system", content: "You are a concise, factual assistant. Always cite source and page numbers." },
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

  const answerCorrectness = await evaluateAnswerCorrectness(
    testItem.question,
    generatedAnswer,
    testItem.expectedAnswer
  );
  const citationCorrectness = evaluateCitationCorrectness(generatedAnswer, testItem.relevantPages);

  return {
    recall,
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
 * 4. BENCHMARK SUITE EXECUTION & REPORTING
 * ============================================================================
 */
async function main() {
  console.log("======================================================================");
  console.log("      COMPREHENSIVE MULTI-DOCUMENT RAG BENCHMARK & ABLATION STUDY     ");
  console.log("======================================================================\n");

  const args = process.argv.slice(2);
  const limitArgIdx = args.indexOf("--limit");
  const categoryArgIdx = args.indexOf("--category");

  let testDataset = EVAL_DATASET;
  if (categoryArgIdx !== -1 && args[categoryArgIdx + 1]) {
    const targetCat = args[categoryArgIdx + 1].toLowerCase();
    testDataset = testDataset.filter((t) => t.category.toLowerCase().includes(targetCat));
  }
  if (limitArgIdx !== -1 && args[limitArgIdx + 1]) {
    const limit = parseInt(args[limitArgIdx + 1], 10);
    if (!isNaN(limit)) testDataset = testDataset.slice(0, limit);
  }

  console.log(`Corpus: documents/48 laws.pdf (651 pages) + documents/cn.pdf (5 pages)`);
  console.log(`Evaluating ${testDataset.length} queries across ${new Set(testDataset.map((d) => d.category)).size} categories.\n`);

  // Load cached multi-document chunks
  console.log("Loading multi-document chunks from cache...");
  const rawData = await readFile("documents/chunks.json", "utf8");
  const chunks = JSON.parse(rawData);
  console.log(`Loaded ${chunks.length} chunks into memory.`);

  console.log("Indexing chunks into high-speed vector store...");
  const embeddings = new OpenAIEmbeddings({
    modelName: "text-embedding-3-small",
    apiKey: process.env.OPENAI_API_KEY,
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
  console.log("Vector store ready.\n");

  const resources = { vectorStore, chunks };

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
      name: "2. + Hybrid (Vector + BM25)",
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

  const overallResults = [];
  const categoryBreakdown = {};

  for (const config of ablationConfigs) {
    console.log(`\n>>> Evaluating [${config.name}] on ${testDataset.length} queries...`);
    const metrics = {
      recalls: [],
      mrrs: [],
      correctnessScores: [],
      citationScores: [],
      latencies: [],
      contextSizes: [],
    };

    let count = 0;
    for (const item of testDataset) {
      count++;
      const res = await runRAGPipeline(item, config, resources);
      metrics.recalls.push(res.recall);
      metrics.mrrs.push(res.mrr);
      metrics.correctnessScores.push(res.answerCorrectness);
      metrics.citationScores.push(res.citationCorrectness);
      metrics.latencies.push(res.latency);
      metrics.contextSizes.push(res.contextChars);

      if (!categoryBreakdown[item.category]) categoryBreakdown[item.category] = {};
      if (!categoryBreakdown[item.category][config.name]) {
        categoryBreakdown[item.category][config.name] = { recalls: [], mrrs: [] };
      }
      categoryBreakdown[item.category][config.name].recalls.push(res.recall);
      categoryBreakdown[item.category][config.name].mrrs.push(res.mrr);

      if (count % 5 === 0 || count === testDataset.length) {
        process.stdout.write(`  [${count}/${testDataset.length} queries completed]\n`);
      }
    }

    const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;

    overallResults.push({
      Configuration: config.name,
      "Recall@3": `${Math.round(avg(metrics.recalls) * 100)}%`,
      MRR: avg(metrics.mrrs).toFixed(3),
      "Answer Correctness": `${Math.round(avg(metrics.correctnessScores))}%`,
      "Citation Accuracy": `${Math.round(avg(metrics.citationScores) * 100)}%`,
      "Avg Latency": `${Math.round(avg(metrics.latencies))}ms`,
      "Context Chars": Math.round(avg(metrics.contextSizes)),
    });
  }

  // Print Overall Ablation Table
  console.log("\n======================================================================");
  console.log("                     OVERALL ABLATION RESULTS TABLE                   ");
  console.log("======================================================================\n");
  console.table(overallResults);

  // Print Category Breakdown Table
  console.log("\n======================================================================");
  console.log("             CATEGORY-BY-CATEGORY RECALL@3 BREAKDOWN                  ");
  console.log("======================================================================\n");

  const catRows = Object.entries(categoryBreakdown).map(([cat, configData]) => {
    const row = { Category: cat };
    ablationConfigs.forEach((c) => {
      const recs = configData[c.name]?.recalls || [];
      const avgRec = recs.length > 0 ? recs.reduce((a, b) => a + b, 0) / recs.length : 0;
      const colName = c.name.split(". ")[1] || c.name;
      row[colName] = `${Math.round(avgRec * 100)}%`;
    });
    return row;
  });

  console.table(catRows);
}

main().catch((err) => {
  console.error("Evaluation run failed:", err);
  process.exit(1);
});
