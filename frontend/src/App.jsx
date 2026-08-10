import { useEffect, useState } from "react";
import { Dna, CircleDot } from "lucide-react";
import FileUpload from "./components/FileUpload.jsx";
import DiffViewer from "./components/DiffViewer.jsx";
import AttentionHeatmap from "./components/AttentionHeatmap.jsx";
import ExportMenu from "./components/ExportMenu.jsx";
import { TooltipProvider } from "./components/ui/tooltip.jsx";
import { checkHealth } from "./api/client.js";

export default function App() {
  const [result, setResult] = useState(null);
  const [noisyInput, setNoisyInput] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [health, setHealth] = useState({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((data) => !cancelled && setHealth(data))
      .catch(() => !cancelled && setHealth({ status: "unreachable", model_loaded: false, device: "unknown" }));
    return () => {
      cancelled = true;
    };
  }, []);

  const handleResult = (response, submittedSequence) => {
    setResult(response);
    setNoisyInput(submittedSequence);
  };

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-void">
        <header className="border-b border-line">
          <div className="mx-auto flex max-w-6xl items-center gap-3 px-6 py-4">
            <Dna className="h-5 w-5 text-primary" aria-hidden="true" />
            <div>
              <h1 className="font-mono text-sm font-semibold uppercase tracking-widest text-ink-primary">
                Genome Repair Console
              </h1>
              <p className="text-xs text-ink-muted">Context-driven neural sequence translation</p>
            </div>
            <HealthIndicator health={health} />
          </div>
        </header>

        <main className="mx-auto flex max-w-6xl flex-col gap-5 px-6 py-6">
          <FileUpload onResult={handleResult} onLoadingChange={setIsLoading} />

          {isLoading && (
            <div className="rounded-lg border border-line bg-surface px-4 py-3 text-sm text-ink-muted">
              <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-primary" /> Running correction --
              long reads are chunked and may take a few moments.
            </div>
          )}

          {result && !isLoading && (
            <div className="flex items-center gap-3">
              <div className="flex-1">
                <MetricsStrip metrics={result.metrics} />
              </div>
              <ExportMenu result={result} />
            </div>
          )}

          <div className="grid gap-5">
            <DiffViewer noisyInput={noisyInput} correctedSequence={result?.corrected_sequence ?? null} />
            <AttentionHeatmap attentionChunks={result?.attention_chunks ?? null} />
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}

function HealthIndicator({ health }) {
  const isConnected = health.status === "ok" && health.model_loaded;
  const dotColor = isConnected ? "text-primary" : health.status === "loading" ? "text-muted-foreground" : "text-destructive";
  const label = isConnected
    ? `connected · ${health.device}`
    : health.status === "loading"
    ? "connecting..."
    : "backend unreachable";

  return (
    <div className="ml-auto flex items-center gap-1.5 font-mono text-xs text-ink-muted">
      <CircleDot className={`h-3 w-3 ${dotColor}`} aria-hidden="true" />
      {label}
    </div>
  );
}

function MetricsStrip({ metrics }) {
  const items = [
    { label: "identity vs input", value: formatPercent(1 - metrics.edit_distance / Math.max(metrics.input_length, 1)) },
    { label: "edit distance", value: metrics.edit_distance },
    { label: "substitutions", value: metrics.num_substitutions },
    { label: "insertions", value: metrics.num_insertions },
    { label: "deletions", value: metrics.num_deletions },
    { label: "chunks", value: metrics.num_chunks },
    { label: "latency", value: `${metrics.latency_ms.toFixed(0)} ms` },
  ];

  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-4 lg:grid-cols-7">
      {items.map((item) => (
        <div key={item.label} className="bg-surface px-3 py-2.5">
          <p className="font-mono text-base font-semibold text-ink-primary">{item.value}</p>
          <p className="text-[10px] uppercase tracking-wide text-ink-muted">{item.label}</p>
        </div>
      ))}
    </div>
  );
}

function formatPercent(fraction) {
  return `${(Math.max(0, fraction) * 100).toFixed(1)}%`;
}
