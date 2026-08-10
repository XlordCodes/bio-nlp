import { useCallback, useRef, useState } from "react";
import { Dna, UploadCloud, Loader2, AlertCircle, FileText } from "lucide-react";
import { correctSequence, correctSequenceFile, ApiError } from "../api/client.js";

const ALLOWED_EXTENSIONS = [".fasta", ".fa"];

/**
 * Mirrors backend/schemas.py's InferenceRequest normalization (strip a
 * leading FASTA header line, drop whitespace/newlines, uppercase) purely
 * for DISPLAY purposes -- the server independently re-validates and
 * normalizes the real input itself. This is only so the UI can diff
 * against the exact string the backend actually corrected, including for
 * file uploads, where the app would otherwise never see the raw text.
 */
function normalizeSequenceText(text) {
  const lines = text.split(/\r?\n/).filter((line) => !line.trim().startsWith(">"));
  return lines.join("").replace(/\s/g, "").toUpperCase();
}

/**
 * FileUpload
 * -----------
 * Two input modes -- paste raw sequence, or drop a single-record .fasta/.fa
 * file -- both funneling into the same submit path. Manages its own
 * request lifecycle (loading, inline error) and reports results/loading
 * state up to the parent, which owns what happens with a successful
 * correction (feeding DiffViewer / AttentionHeatmap).
 */
export default function FileUpload({ onResult, onLoadingChange }) {
  const [mode, setMode] = useState("paste"); // "paste" | "file"
  const [sequenceText, setSequenceText] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const fileInputRef = useRef(null);

  const setLoading = useCallback(
    (value) => {
      setIsSubmitting(value);
      onLoadingChange?.(value);
    },
    [onLoadingChange]
  );

  const validateAndSetFile = (file) => {
    if (!file) return;
    const lowerName = file.name.toLowerCase();
    const hasAllowedExtension = ALLOWED_EXTENSIONS.some((ext) => lowerName.endsWith(ext));
    if (!hasAllowedExtension) {
      setErrorMessage(`Unsupported file type. Expected one of: ${ALLOWED_EXTENSIONS.join(", ")}`);
      return;
    }
    setErrorMessage(null);
    setSelectedFile(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    validateAndSetFile(file);
  };

  const handleSubmit = async () => {
    setErrorMessage(null);

    if (mode === "paste" && sequenceText.trim().length === 0) {
      setErrorMessage("Paste a nucleotide sequence before running correction.");
      return;
    }
    if (mode === "file" && !selectedFile) {
      setErrorMessage("Choose or drop a .fasta/.fa file before running correction.");
      return;
    }

    setLoading(true);
    try {
      const result =
        mode === "paste" ? await correctSequence(sequenceText) : await correctSequenceFile(selectedFile);
      const submittedSequence =
        mode === "paste" ? normalizeSequenceText(sequenceText) : normalizeSequenceText(await selectedFile.text());
      onResult(result, submittedSequence);
    } catch (err) {
      const message =
        err instanceof ApiError
          ? `${err.message}: ${err.detail}`
          : "Could not reach the correction service. Check that the backend is running.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-lg border border-line bg-surface shadow-panel">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <Dna className="h-4 w-4 text-primary" aria-hidden="true" />
        <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted">Input</h2>

        <div className="ml-auto flex gap-1 rounded-md bg-void p-1">
          <ModeButton active={mode === "paste"} onClick={() => setMode("paste")} label="Paste sequence" />
          <ModeButton active={mode === "file"} onClick={() => setMode("file")} label="Upload FASTA" />
        </div>
      </div>

      <div className="p-4">
        {mode === "paste" ? (
          <textarea
            value={sequenceText}
            onChange={(e) => setSequenceText(e.target.value)}
            placeholder="Paste a raw nucleotide sequence (A/C/G/T/N), or a FASTA block with a header line..."
            rows={6}
            spellCheck={false}
            className="w-full resize-y rounded-md border border-line bg-void px-3 py-2 font-mono text-sm text-ink-primary placeholder:text-ink-faint focus:border-primary"
          />
        ) : (
          <div
            role="button"
            tabIndex={0}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed px-4 py-10 text-center transition-colors duration-200 ${
              isDragging ? "border-primary bg-primary/5" : "border-line hover:border-ink-faint"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".fasta,.fa"
              className="hidden"
              onChange={(e) => validateAndSetFile(e.target.files?.[0])}
            />
            {selectedFile ? (
              <>
                <FileText className="h-6 w-6 text-primary" aria-hidden="true" />
                <p className="font-mono text-sm text-ink-primary">{selectedFile.name}</p>
                <p className="text-xs text-ink-muted">Click to choose a different file</p>
              </>
            ) : (
              <>
                <UploadCloud className="h-6 w-6 text-ink-muted" aria-hidden="true" />
                <p className="text-sm text-ink-primary">Drop a .fasta or .fa file here, or click to browse</p>
                <p className="text-xs text-ink-muted">Single-record files only, one read per upload</p>
              </>
            )}
          </div>
        )}

        {errorMessage && (
          <div className="mt-3 flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
            <span>{errorMessage}</span>
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={isSubmitting}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-md bg-primary px-4 py-2.5 font-medium text-primary-foreground transition-colors duration-200 hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              Correcting sequence...
            </>
          ) : (
            "Run correction"
          )}
        </button>
      </div>
    </div>
  );
}

function ModeButton({ active, onClick, label }) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-2.5 py-1 text-xs font-medium transition-colors duration-200 cursor-pointer ${
        active ? "bg-surface-raised text-ink-primary" : "text-ink-muted hover:text-ink-primary"
      }`}
    >
      {label}
    </button>
  );
}
