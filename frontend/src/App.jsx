import { useState, useEffect, useCallback, useRef } from 'react';
import { runQuery, healthCheck } from './api';

/* ─── Premium Vector SVGs ──────────────────────────────── */
const LogoIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="url(#logoGrad)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <defs>
      <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stopColor="var(--color-primary)" />
        <stop offset="100%" stopColor="var(--color-accent)" />
      </linearGradient>
    </defs>
    <path d="M12 2L2 12l10 10 10-10L12 2z" />
    <path d="M12 6L6 12l6 6 6-6-6-6z" opacity="0.5" />
    <circle cx="12" cy="12" r="1.5" fill="url(#logoGrad)" />
  </svg>
);

const LockIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
  </svg>
);

const SearchIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
  </svg>
);

const DocumentIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
  </svg>
);

const BrainIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.44 2.5 2.5 0 0 1 0-3.12 3.02 3.02 0 0 1 0-3.88 2.5 2.5 0 0 1 0-3.12A2.5 2.5 0 0 1 9.5 2z"/>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.44 2.5 2.5 0 0 0 0-3.12 3.02 3.02 0 0 0 0-3.88 2.5 2.5 0 0 0 0-3.12A2.5 2.5 0 0 0 14.5 2z"/>
  </svg>
);

const CheckIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);

const ChartIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"/>
    <line x1="12" y1="20" x2="12" y2="4"/>
    <line x1="6" y1="20" x2="6" y2="14"/>
  </svg>
);

const CodeIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
  </svg>
);

const SparklesIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-accent)' }}>
    <path d="M12 3v1M12 20v1M3 12h1M20 12h1M5.9 5.9l.7.7M17.4 17.4l.7.7M5.9 17.4l.7-.7M17.4 5.9l.7-.7"/>
  </svg>
);

const SparkleFilled = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" style={{ color: 'var(--color-primary)' }}>
    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
  </svg>
);

const AlertIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ color: 'var(--color-danger)' }}>
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
    <line x1="12" y1="9" x2="12" y2="13"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);

/* ─── Helpers ──────────────────────────────────────────── */
function formatTime(seconds) {
  if (seconds == null) return '—';
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(2)}s`;
}

function truncate(str, len = 180) {
  if (!str) return '';
  return str.length > len ? str.slice(0, len) + '…' : str;
}

function heatColor(value) {
  // 0 -> Slate/Black, 1 -> Luminous Cyan
  const clamped = Math.max(0, Math.min(1, value));
  const r = Math.round(9 + clamped * (6 - 9));
  const g = Math.round(9 + clamped * (182 - 9));
  const b = Math.round(14 + clamped * (212 - 14));
  return `rgb(${r},${g},${b})`;
}

function claimStatusBadgeClass(status) {
  const s = (status || '').toLowerCase();
  if (s === 'supported' || s === 'verified') return 'badge-supported';
  if (s === 'unsupported' || s === 'refuted') return 'badge-unsupported';
  return 'badge-uncertain';
}

const TABS = [
  { key: 'evidence', label: 'Evidence', icon: <DocumentIcon /> },
  { key: 'verification', label: 'Verification', icon: <BrainIcon /> },
  { key: 'claims', label: 'Fact Check', icon: <CheckIcon /> },
  { key: 'agents', label: 'Agent Log', icon: <CodeIcon /> },
  { key: 'queries', label: 'Queries', icon: <SearchIcon /> },
];

/* ═══════════════════════════════════════════════════════════
   Main App Component
   ═══════════════════════════════════════════════════════════ */
export default function App() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('evidence');
  const [queryHistory, setQueryHistory] = useState([]);
  const [healthStatus, setHealthStatus] = useState('checking');
  const [pipelineStep, setPipelineStep] = useState(0); // 0 -> idle, 1-7 active steps
  const textareaRef = useRef(null);

  const checkHealth = useCallback(async () => {
    try {
      await healthCheck();
      setHealthStatus('online');
    } catch {
      setHealthStatus('offline');
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const handleSubmit = useCallback(async () => {
    const q = question.trim();
    if (!q || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResult(null);
    setPipelineStep(1);

    // Simulate pipeline step advancement during the API wait
    const intervals = [
      setTimeout(() => setPipelineStep(2), 1500),
      setTimeout(() => setPipelineStep(3), 3000),
      setTimeout(() => setPipelineStep(4), 8000),
      setTimeout(() => setPipelineStep(5), 14000),
      setTimeout(() => setPipelineStep(6), 18000),
      setTimeout(() => setPipelineStep(7), 22000),
    ];

    try {
      const data = await runQuery(q);
      intervals.forEach(clearTimeout);
      setPipelineStep(8); // Completed
      setResult(data);
      setActiveTab('evidence');
      setQueryHistory((prev) => {
        const next = [q, ...prev.filter((p) => p !== q)];
        return next.slice(0, 5);
      });
    } catch (err) {
      intervals.forEach(clearTimeout);
      setPipelineStep(0);
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setIsLoading(false);
    }
  }, [question, isLoading]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const selectHistory = useCallback((q) => {
    setQuestion(q);
    if (textareaRef.current) textareaRef.current.focus();
  }, []);

  const metrics = result?.pipeline_metrics || {};
  const allDocs = result?.documents || [];
  const usefulDocs = allDocs.filter((d) => d.is_useful);
  const verification = result?.verification;
  const consistency = verification?.consistency_score;

  return (
    <>
      {/* ── Sidebar ──────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">
            <LogoIcon />
          </div>
          <div>
            <h1>OMNITRUST-RAG</h1>
            <p>Verification Core</p>
          </div>
        </div>

        <textarea
          ref={textareaRef}
          className="query-input"
          placeholder="Ask a question or enter a hypothesis to verify..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        
        <button
          className="btn btn-primary btn-block"
          onClick={handleSubmit}
          disabled={isLoading || !question.trim()}
        >
          {isLoading ? (
            <>
              <span className="spinner" style={{ width: '16px', height: '16px' }} />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <SearchIcon />
              <span>Verify Claim</span>
            </>
          )}
        </button>
        <span className="text-muted" style={{ fontSize: '0.7rem', marginTop: '-8px', textAlign: 'center' }}>
          Press Ctrl + Enter to submit
        </span>

        {queryHistory.length > 0 && (
          <div className="history-section">
            <h4>History</h4>
            {queryHistory.map((q, i) => (
              <div key={i} className="history-item" onClick={() => selectHistory(q)} title={q}>
                {q}
              </div>
            ))}
          </div>
        )}

        <div className="health-indicator">
          <span className={`health-dot ${healthStatus}`} />
          <span>Pipeline: {healthStatus === 'online' ? 'Online' : healthStatus === 'offline' ? 'Offline' : 'Connecting...'}</span>
        </div>
      </aside>

      {/* ── Main Content ─────────────────────────────────── */}
      <main className="main-content">
        {error && (
          <div className="error-banner">
            <span className="error-banner-icon"><AlertIcon /></span>
            <div className="error-banner-body">
              <div className="error-banner-title">Verification Pipeline Failed</div>
              <div className="error-banner-msg">{error}</div>
            </div>
            <button className="error-dismiss" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* Pipeline Progress Visualizer */}
        {isLoading && (
          <div className="pipeline-visualizer">
            <div className="pipeline-title">Active Verification Sequence</div>
            <div className="pipeline-flow">
              <PipelineStepNode index={1} active={pipelineStep === 1} completed={pipelineStep > 1} label="Plan" icon={<SearchIcon />} />
              <div className={`pipeline-line ${pipelineStep > 1 ? 'completed' : ''} ${pipelineStep === 2 ? 'active' : ''}`} />
              <PipelineStepNode index={2} active={pipelineStep === 2} completed={pipelineStep > 2} label="Strategy" icon={<BrainIcon />} />
              <div className={`pipeline-line ${pipelineStep > 2 ? 'completed' : ''} ${pipelineStep === 3 ? 'active' : ''}`} />
              <PipelineStepNode index={3} active={pipelineStep === 3} completed={pipelineStep > 3} label="Retrieve" icon={<DocumentIcon />} />
              <div className={`pipeline-line ${pipelineStep > 3 ? 'completed' : ''} ${pipelineStep === 4 ? 'active' : ''}`} />
              <PipelineStepNode index={4} active={pipelineStep === 4} completed={pipelineStep > 4} label="Score" icon={<ChartIcon />} />
              <div className={`pipeline-line ${pipelineStep > 4 ? 'completed' : ''} ${pipelineStep === 5 ? 'active' : ''}`} />
              <PipelineStepNode index={5} active={pipelineStep === 5} completed={pipelineStep > 5} label="Verify" icon={<LockIcon />} />
              <div className={`pipeline-line ${pipelineStep > 5 ? 'completed' : ''} ${pipelineStep === 6 ? 'active' : ''}`} />
              <PipelineStepNode index={6} active={pipelineStep === 6} completed={pipelineStep > 6} label="Critic" icon={<CodeIcon />} />
              <div className={`pipeline-line ${pipelineStep > 6 ? 'completed' : ''} ${pipelineStep === 7 ? 'active' : ''}`} />
              <PipelineStepNode index={7} active={pipelineStep === 7} completed={pipelineStep > 7} label="Synthesize" icon={<SparklesIcon />} />
            </div>
          </div>
        )}

        {isLoading && (
          <div className="loading-overlay">
            <div className="spinner" style={{ width: '48px', height: '48px' }} />
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', fontWeight: 500 }}>
              Orchestrating multi-agent consensus verification...
            </p>
          </div>
        )}

        {!isLoading && !result && !error && <EmptyState />}

        {!isLoading && result && (
          <>
            {/* Metrics Dashboard */}
            <div className="metrics-bar">
              <MetricCard
                label="Total Latency"
                value={formatTime(metrics.total_pipeline_s)}
                variant="primary"
                sub="end-to-end"
              />
              <MetricCard
                label="Evidence Retrieved"
                value={allDocs.length}
                variant="accent"
                sub={`${usefulDocs.length} useful, ${allDocs.length - usefulDocs.length} filtered`}
              />
              <MetricCard
                label="Consensus Agreement"
                value={consistency != null ? (consistency * 100).toFixed(1) + '%' : '—'}
                variant={consistency >= 0.7 ? 'success' : consistency >= 0.4 ? 'warning' : 'warning'}
                sub="divergence audit"
              />
              <MetricCard
                label="Retrieval Phase"
                value={formatTime(metrics.step3_retrieval_s)}
                variant="accent"
                sub="hybrid + nvidia rerank"
              />
              <MetricCard
                label="Verification Phase"
                value={formatTime(metrics.step7_verification_s)}
                variant="primary"
                sub="4-head divergence"
              />
              <MetricCard
                label="Synthesizer"
                value={formatTime(metrics.step9_synthesis_s)}
                variant="success"
                sub="cited answer"
              />
            </div>

            {/* Answer Card */}
            <div className="answer-card">
              <div className="answer-card-header">
                <SparklesIcon />
                <h2>Verified Consensus Synthesis</h2>
              </div>
              <div className="answer-card-body">
                {result.final_answer.split('\n').map((line, idx) => {
                  if (line.startsWith('## ')) {
                    return <h2 key={idx}>{line.replace('## ', '')}</h2>;
                  }
                  if (line.startsWith('* ') || line.startsWith('- ')) {
                    return <li key={idx} style={{ marginLeft: '16px', marginBottom: '8px' }}>{line.substring(2)}</li>;
                  }
                  if (line.trim() === '') return <div key={idx} style={{ height: '8px' }} />;
                  return <p key={idx}>{line}</p>;
                })}
              </div>
            </div>

            {/* Tabs Navigation */}
            <nav className="tab-nav">
              {TABS.map((tab) => (
                <button
                  key={tab.key}
                  className={`tab-btn${activeTab === tab.key ? ' active' : ''}`}
                  onClick={() => setActiveTab(tab.key)}
                >
                  {tab.icon}
                  <span>{tab.label}</span>
                </button>
              ))}
            </nav>

            {/* Tab Panels */}
            <div className="tab-panel">
              {activeTab === 'evidence' && <EvidencePanel documents={allDocs} />}
              {activeTab === 'verification' && <VerificationPanel verification={verification} />}
              {activeTab === 'claims' && <ClaimsPanel claims={result.checked_claims} />}
              {activeTab === 'agents' && <AgentLogPanel logs={result.agent_logs} />}
              {activeTab === 'queries' && (
                <QueriesPanel
                  queries={result.queries_used}
                  decision={result.strategist_decision}
                />
              )}
            </div>
          </>
        )}
      </main>
    </>
  );
}

/* ─── Pipeline Step Node ────────────────────────────────── */
function PipelineStepNode({ active, completed, label, icon }) {
  return (
    <div className={`pipeline-step ${active ? 'active' : ''} ${completed ? 'completed' : ''}`}>
      <div className="pipeline-node">
        {completed ? <CheckIcon /> : icon}
      </div>
      <div className="pipeline-label">{label}</div>
    </div>
  );
}

/* ─── MetricCard ────────────────────────────────────────── */
function MetricCard({ label, value, variant, sub }) {
  return (
    <div className="metric-card">
      <span className="metric-label">{label}</span>
      <span className={`metric-value ${variant}`}>{value}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  );
}

/* ─── EmptyState ────────────────────────────────────────── */
function EmptyState() {
  return (
    <div className="empty-state">
      <div style={{ color: 'var(--color-primary)', marginBottom: '16px' }}>
        <LogoIcon />
      </div>
      <h2>Secure Multi-Agent Evidence Verification</h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
        Type a research topic or claims-based hypothesis into the sidebar input to execute the 10-step OmniTrust-RAG consensus pipeline.
      </p>

      <div className="empty-features">
        <div className="empty-feature">
          <div className="empty-feature-icon"><SearchIcon /></div>
          <h4>Strategic Retrieval</h4>
          <p>Planner and Strategist collaborate to construct Wikipedia and local queries, reranking results via cloud NVIDIA NeMo.</p>
        </div>
        <div className="empty-feature">
          <div className="empty-feature-icon"><LogoIcon /></div>
          <h4>Source Independence</h4>
          <p>Scoring engine clusters sources using TF-IDF cosine similarity to penalize duplicate or non-independent evidence.</p>
        </div>
        <div className="empty-feature">
          <div className="empty-feature-icon"><BrainIcon /></div>
          <h4>Family Attention</h4>
          <p>Lateral similarity matrix analyzes divergence across Semantic, Named Entity, Temporal, and Context heads.</p>
        </div>
        <div className="empty-feature">
          <div className="empty-feature-icon"><CheckIcon /></div>
          <h4>Meticulous Fact Check</h4>
          <p>Critic agent audits every individual claim against retrieved documents to assert verified status before final answer generation.</p>
        </div>
      </div>
    </div>
  );
}

/* ─── Evidence Panel ────────────────────────────────────── */
function EvidencePanel({ documents }) {
  if (!documents || documents.length === 0) {
    return <p className="text-muted" style={{ padding: 12 }}>No evidence documents were retrieved.</p>;
  }

  return (
    <div>
      <div className="section-header">
        <h3>Retrieved Documents</h3>
        <span className="count">{documents.length}</span>
      </div>
      <div className="evidence-grid">
        {documents.map((doc, i) => (
          <div key={doc.id || i} className={`evidence-card${doc.is_duplicate ? ' duplicate' : ''}`}>
            <div className="evidence-card-header">
              <div>
                <div className="evidence-card-title">{doc.title}</div>
                {doc.source && <div className="evidence-card-source">{doc.source}</div>}
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                {doc.is_duplicate && <span className="badge badge-duplicate">Duplicate</span>}
                {doc.is_useful ? (
                  <span className="badge badge-useful">Useful</span>
                ) : (
                  <span className="badge badge-not-useful">Filtered</span>
                )}
              </div>
            </div>
            <div className="evidence-card-text">{truncate(doc.text, 250)}</div>
            <div className="evidence-scores">
              <span className="pill pill-primary">Independence: {(doc.independence_score * 100).toFixed(0)}%</span>
              <span className="pill pill-accent">Utility: {(doc.utility_score * 100).toFixed(0)}%</span>
              {doc.novelty != null && <span className="pill pill-success">Novelty: {(doc.novelty * 100).toFixed(0)}%</span>}
              {doc.contradiction != null && doc.contradiction > 0 && <span className="pill pill-danger">Contradiction: {(doc.contradiction * 100).toFixed(0)}%</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Verification Panel ────────────────────────────────── */
function VerificationPanel({ verification }) {
  if (!verification) {
    return <p className="text-muted" style={{ padding: 12 }}>No verification analysis data available.</p>;
  }

  const { consistency_score, js_divergence, head_perspectives, similarity_matrix } = verification;

  return (
    <div className="verification-section">
      <div className="glass-card">
        <div className="section-header">
          <h3>Jensen–Shannon Divergence (Head Agreement)</h3>
          <span className="pill pill-primary">Value: {js_divergence != null ? js_divergence.toFixed(4) : '—'}</span>
        </div>
        <div className="divergence-meter">
          <div className="divergence-bar-track">
            <div
              className="divergence-bar-fill"
              style={{ width: `${Math.min((js_divergence || 0) * 100, 100)}%` }}
            />
          </div>
          <div className="divergence-labels">
            <span>0.0 — Complete Agreement</span>
            <span>1.0 — Maximal Divergence</span>
          </div>
        </div>
      </div>

      {head_perspectives && head_perspectives.length > 0 && (
        <div className="glass-card">
          <div className="section-header">
            <h3>Head Perspective Distribution</h3>
          </div>
          <div className="head-perspectives">
            {head_perspectives.map((head, i) => {
              const total = (head.supported || 0) + (head.uncertain || 0) + (head.unsupported || 0);
              const pctS = total ? (head.supported / total) * 100 : 0;
              const pctU = total ? (head.uncertain / total) * 100 : 0;
              const pctN = total ? (head.unsupported / total) * 100 : 0;

              return (
                <div key={head.head_id ?? i} className="head-row">
                  <div className="head-label">
                    <span className="head-label-name">{head.name}</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                      Support: {head.supported} | Conflicting: {head.unsupported}
                    </span>
                  </div>
                  <div className="head-bar-track">
                    <div className="head-bar-seg supported" style={{ width: `${pctS}%` }} />
                    <div className="head-bar-seg uncertain" style={{ width: `${pctU}%` }} />
                    <div className="head-bar-seg unsupported" style={{ width: `${pctN}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {similarity_matrix && similarity_matrix.length > 0 && (
        <div className="glass-card">
          <div className="section-header">
            <h3>Lateral Head Similarity Matrix</h3>
          </div>
          <div className="heatmap-container">
            <div
              className="heatmap-grid"
              style={{ gridTemplateColumns: `repeat(${similarity_matrix[0].length}, 42px)` }}
            >
              {similarity_matrix.flatMap((row, ri) =>
                row.map((val, ci) => (
                  <div
                    key={`${ri}-${ci}`}
                    className="heatmap-cell"
                    style={{ backgroundColor: heatColor(val) }}
                    title={`Similarity [Head ${ri} ⟷ Head ${ci}]: ${val.toFixed(4)}`}
                  >
                    {val.toFixed(2)}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── Claims Panel ──────────────────────────────────────── */
function ClaimsPanel({ claims }) {
  if (!claims || claims.length === 0) {
    return <p className="text-muted" style={{ padding: 12 }}>No fact-checked claims extracted.</p>;
  }

  return (
    <div>
      <div className="section-header">
        <h3>Critic Claims Cross-Verification</h3>
        <span className="count">{claims.length}</span>
      </div>
      <div className="claims-list">
        {claims.map((c, i) => (
          <div key={i} className="claim-card">
            <div className="claim-text">"{c.claim}"</div>
            <div className="claim-meta">
              <span className={`badge ${claimStatusBadgeClass(c.status)}`}>{c.status.toUpperCase()}</span>
              {c.evidence_doc_id && <span className="pill">Doc Ref: {c.evidence_doc_id}</span>}
            </div>
            {c.reason && <div className="claim-reason">{c.reason}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Agent Log Panel ───────────────────────────────────── */
function AgentLogPanel({ logs }) {
  if (!logs || logs.length === 0) {
    return <p className="text-muted" style={{ padding: 12 }}>No agent execution logs available.</p>;
  }

  return (
    <div>
      <div className="section-header">
        <h3>Agent Execution Trace</h3>
        <span className="count">{logs.length} events</span>
      </div>
      <div className="agent-timeline">
        {logs.map((log, i) => {
          const sender = log.sender || 'System';
          const payloadStr = typeof log.payload === 'string' ? log.payload : JSON.stringify(log.payload, null, 2);

          return (
            <div key={i} className="agent-log-card" data-role={sender}>
              <div className="agent-log-meta">
                <span className={`agent-sender-badge ${sender}`}>{sender}</span>
                <span className="agent-msg-type">{log.msg_type}</span>
              </div>
              <div className="agent-log-payload">
                {payloadStr.length > 500 ? (
                  <details>
                    <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)' }}>
                      Show Details ({payloadStr.substring(0, 80)}...)
                    </summary>
                    <pre style={{ marginTop: '8px', whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>
                      {payloadStr}
                    </pre>
                  </details>
                ) : (
                  <pre style={{ whiteSpace: 'pre-wrap', color: 'var(--text-secondary)' }}>{payloadStr}</pre>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Queries Panel ─────────────────────────────────────── */
function QueriesPanel({ queries, decision }) {
  return (
    <div>
      <div className="section-header">
        <h3>Orchestrated Search Queries</h3>
        {queries && <span className="count">{queries.length}</span>}
        {decision && (
          <span className={`badge badge-${decision === 'approve' ? 'approve' : 'modify'}`} style={{ marginLeft: 'auto' }}>
            Decision: {decision.toUpperCase()}
          </span>
        )}
      </div>
      {queries && queries.length > 0 ? (
        <div className="queries-list">
          {queries.map((q, i) => (
            <div key={i} className="query-item">
              <span className="query-index">{i + 1}</span>
              <span style={{ color: 'var(--text-primary)' }}>{q}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-muted" style={{ padding: 12 }}>No queries planned.</p>
      )}
    </div>
  );
}
