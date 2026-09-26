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
 * Reciprocal Rank Fusion (RRF)
 * Merges multiple ranked lists into a unified list.
 *
 * Formula: RRF_score(d) = Σ [ weight_i / (rank_i(d) + k) ]
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

  return Array.from(docMap.values())
    .sort((a, b) => b.score - a.score)
    .map((item) => item.document);
}

/**
 * LLM Cross-Encoder Reranker
 * Evaluates candidate chunks against the user query, scoring each on a 0-100 relevance scale.
 *
 * @param {string} query - The user query
 * @param {Array<Document>} candidates - Up to 50 candidates retrieved via hybrid search
 * @param {number} topK - Number of top documents to return after reranking
 * @returns {Promise<Array<Document>>} Top K reranked documents
 */
async function rerankCandidates(query, candidates, topK = 5) {
  if (!candidates || candidates.length === 0) return [];

  console.log(`\n--- Reranker Stage ---`);
  console.log(`Evaluating ${candidates.length} candidates against query: "${query}"...`);

  const candidateExcerpts = candidates
    .map((doc, idx) => {
      const page = doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? "?";
      // Truncate to reasonable length to conserve tokens while preserving context
      const content = doc.pageContent.replace(/\s+/g, " ").trim().slice(0, 500);
      return `[Candidate ID ${idx}] (Page ${page}):\n${content}`;
    })
    .join("\n\n");

  const prompt = `
You are an expert search relevance evaluator (Cross-Encoder Reranker).
User Query: "${query}"

Your task:
Carefully evaluate each candidate passage and score its relevance to answering the user query.
Assign an integer score between 0 and 100:
- 90-100: Direct, comprehensive answer to the user query.
- 60-89: Highly relevant context or partial answer.
- 20-59: Tangentially related topic or keyword match with little answer value.
- 0-19: Completely irrelevant or wrong topic.

Respond ONLY with a JSON object in this exact schema:
{
  "scores": [
    { "id": 0, "score": 95, "reason": "Direct explanation of..." },
    { "id": 1, "score": 10, "reason": "Unrelated topic" }
  ]
}

Candidates to evaluate:
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

    // Attach rerank scores and sort descending
    const scoredCandidates = candidates.map((doc, idx) => {
      const evaluation = scoreMap.get(idx) || { score: 0, reason: "No score assigned" };
      return {
        doc,
        rerankScore: evaluation.score,
        reason: evaluation.reason,
      };
    });

    scoredCandidates.sort((a, b) => b.rerankScore - a.rerankScore);

    console.log(`Reranking completed. Top ${Math.min(topK, scoredCandidates.length)} results:`);
    scoredCandidates.slice(0, topK).forEach((item, i) => {
      const page = item.doc.metadata?.pageNumber ?? item.doc.metadata?.loc?.pageNumber;
      console.log(
        `#${i + 1} | Score: ${item.rerankScore}/100 | Page: ${page} | Reason: ${item.reason}`
      );
    });

    return scoredCandidates.slice(0, topK).map((item) => item.doc);
  } catch (error) {
    console.warn("⚠️ Reranker error, falling back to original RRF candidate order:", error.message);
    return candidates.slice(0, topK);
  }
}

async function query(userQuery) {
  console.log(`\n==================================================`);
  console.log(`Query: "${userQuery}"`);
  console.log(`==================================================\n`);

  // --- 1. Vector Search (Semantic) Setup ---
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

  // Pull candidate pool (up to 50)
  const vectorStoreRetriever = vectorStore.asRetriever({ k: 50 });

  // --- 2. BM25 Search (Keyword) Setup ---
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

  // --- 3. Parallel Hybrid Retrieval ---
  console.log("Executing Hybrid Retrieval (Vector + BM25)...");
  const [vectorResults, bm25Results] = await Promise.all([
    vectorStoreRetriever.invoke(userQuery),
    bm25Retriever.invoke(userQuery),
  ]);

  console.log(`- Vector search returned: ${vectorResults.length} chunks`);
  console.log(`- BM25 search returned: ${bm25Results.length} chunks`);

  // --- 4. Fusion (RRF) -> Candidate Pool (up to 50 candidates) ---
  const fusedCandidates = reciprocalRankFusion(
    [vectorResults, bm25Results],
    60,
    [0.5, 0.5]
  );
  const candidatePool50 = fusedCandidates.slice(0, 50);
  console.log(`Total candidate pool after deduplicated RRF: ${candidatePool50.length} chunks`);

  // --- 5. Reranker Stage -> Top 5 ---
  const top5Docs = await rerankCandidates(userQuery, candidatePool50, 5);

  // --- 6. Generation with Final LLM ---
  const SYSTEM_PROMPT = `
    You are an expert in answering user query based on the provided context about document.
    Do not answer anything beyond what is provided.

    Always also answer the user concisely and state which page number that content is available on and the name/source of the document.

    User Documents:
    ${top5Docs
      .map((e) =>
        JSON.stringify({
          bookName: e.metadata?.source,
          pageContent: e.pageContent,
          pageNumber: e.metadata?.pageNumber ?? e.metadata?.loc?.pageNumber,
        })
      )
      .join("\n\n")}
  `;

  const llmResponse = await openai.chat.completions.create({
    model: "gpt-6-luna",
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: userQuery },
    ],
  });

  console.log(`\n==================================================`);
  console.log(`Final LLM Response:\n`, llmResponse.choices[0].message.content);
  console.log(`==================================================\n`);
  return llmResponse.choices[0].message.content;
}

query("What is the main topic of the document page 2?");
