export function reciprocalRankFusion(
  rankingsList,
  k = 60,
  weights = [0.5, 0.5],
) {
  const docMap = new Map(); // key -> { document, score, ranks }

  rankingsList.forEach((ranking, listIdx) => {
    const weight = weights[listIdx] ?? 1.0;

    ranking.forEach((doc, rankIdx) => {
      const rank = rankIdx + 1; // 1-based rank (1st, 2nd, 3rd, ...)
      const rrfScore = weight / (rank + k);

      // Create a unique key for deduplication based on content and source metadata
      const source = doc.metadata?.source || "unknown";
      const page =
        doc.metadata?.pageNumber ?? doc.metadata?.loc?.pageNumber ?? 0;
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
