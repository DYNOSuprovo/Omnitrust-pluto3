export async function runQuery(question) {
  const res = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const errorBody = await res.text().catch(() => '');
    throw new Error(errorBody || `Query failed with status ${res.status}`);
  }
  return res.json();
}

export async function getCorpus() {
  const res = await fetch('/api/corpus');
  if (!res.ok) throw new Error('Failed to fetch corpus');
  return res.json();
}

export async function addDocuments(documents) {
  const res = await fetch('/api/corpus', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents }),
  });
  if (!res.ok) throw new Error('Failed to add documents');
  return res.json();
}

export async function healthCheck() {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}
