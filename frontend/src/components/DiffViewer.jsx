import { useMemo, useRef, useState } from "react";
import { diffChars } from "diff";
import { FileSearch } from "lucide-react";
import Minimap from "./Minimap.jsx";
import CodonTrack, { FrameSelector } from "./CodonTrack.jsx";

const BASE_COLOR_CLASS = { A: "text-base-A", T: "text-base-T", G: "text-base-G", C: "text-base-C" };
const MAX_RENDERED_LENGTH = 4000; // keep the DOM (one span per base) from becoming unwieldy on huge reads

function baseColorClass(char) {
  return BASE_COLOR_CLASS[char.toUpperCase()] ?? "text-base-N";
}

/** Builds a ruler line (e.g. "0    10   20   30") whose character offsets line up with the monospace sequence below it. */
function buildRuler(length, tickInterval = 10) {
  const chars = new Array(length).fill(" ");
  for (let pos = 0; pos < length; pos += tickInterval) {
    const label = String(pos);
    for (let i = 0; i < label.length && pos + i < length; i++) {
      chars[pos + i] = label[i];
    }
  }
  return chars.join("");
}

/**
 * DiffViewer
 * -----------
 * Renders the corrected sequence character-by-character, colored by
 * nucleotide identity, with insertions/deletions relative to the original
 * noisy input highlighted inline -- the "Biological Correction Ledger"
 * from the design brief. Diffing is character-level Myers diff (the `diff`
 * package), which is efficient enough for read-length sequences; very long
 * sequences are truncated for DISPLAY only (never for the underlying data)
 * to keep the DOM from rendering thousands of individual spans.
 *
 * Also hosts the Minimap (overview scrollbar) and CodonTrack (reading-frame
 * row), both sharing this component's scroll container so all three stay
 * trivially aligned/synced without a separate cross-component scroll bridge.
 */
export default function DiffViewer({ noisyInput, correctedSequence }) {
  const scrollContainerRef = useRef(null);
  const [frameOffset, setFrameOffset] = useState(0);

  const parts = useMemo(() => {
    if (!noisyInput || !correctedSequence) return null;
    return diffChars(noisyInput, correctedSequence);
  }, [noisyInput, correctedSequence]);

  if (!parts) {
    return (
      <Panel>
        <EmptyState />
      </Panel>
    );
  }

  const displayParts = truncateParts(parts, MAX_RENDERED_LENGTH);
  const wasTruncated = displayParts.truncated;
  const rulerString = buildRuler(displayParts.renderedLength);

  return (
    <Panel frameSelector={<FrameSelector frameOffset={frameOffset} onChange={setFrameOffset} />}>
      <div className="flex flex-wrap items-center gap-4 border-b border-line px-4 py-2 text-xs">
        <LegendSwatch colorClass="text-base-A" label="A" />
        <LegendSwatch colorClass="text-base-T" label="T" />
        <LegendSwatch colorClass="text-base-G" label="G" />
        <LegendSwatch colorClass="text-base-C" label="C" />
        <span className="mx-1 h-3 w-px bg-line" />
        <span className="flex items-center gap-1.5 text-ink-muted">
          <span className="rounded-sm bg-primary/20 px-1 text-primary underline decoration-2">inserted</span>
        </span>
        <span className="flex items-center gap-1.5 text-ink-muted">
          <span className="text-destructive/70 line-through decoration-2">removed</span>
        </span>
        <span className="mx-1 h-3 w-px bg-line" />
        <span className="flex items-center gap-1.5 text-ink-muted">
          <span className="rounded-sm bg-destructive/40 px-1 text-destructive">stop</span>
          codon (current frame)
        </span>
      </div>

      <Minimap parts={displayParts.parts} renderedLength={displayParts.renderedLength} scrollContainerRef={scrollContainerRef} />

      <div ref={scrollContainerRef} className="overflow-x-auto px-4 py-3">
        <div className="min-w-max font-mono text-xs leading-relaxed text-ink-faint">{rulerString}</div>
        <div className="min-w-max whitespace-pre font-mono text-sm leading-relaxed">
          {displayParts.parts.map((part, partIndex) =>
            [...part.value].map((char, charIndex) => (
              <span
                key={`${partIndex}-${charIndex}`}
                className={
                  part.removed
                    ? `${baseColorClass(char)} opacity-60 line-through decoration-2 decoration-destructive`
                    : part.added
                    ? `${baseColorClass(char)} bg-primary/20 underline decoration-2 decoration-primary`
                    : baseColorClass(char)
                }
                title={part.removed ? "Removed by correction" : part.added ? "Inserted by correction" : undefined}
              >
                {char}
              </span>
            ))
          )}
        </div>
        <div className="mt-1 border-t border-line/50 pt-1">
          <CodonTrack parts={displayParts.parts} frameOffset={frameOffset} />
        </div>
        {wasTruncated && (
          <p className="mt-2 text-xs text-ink-muted">
            Display truncated to the first {MAX_RENDERED_LENGTH.toLocaleString()} characters. Full corrected
            sequence is available in the exported result.
          </p>
        )}
      </div>
    </Panel>
  );
}

function truncateParts(parts, maxLength) {
  let renderedLength = 0;
  const kept = [];
  for (const part of parts) {
    if (renderedLength >= maxLength) break;
    const remaining = maxLength - renderedLength;
    if (part.value.length <= remaining) {
      kept.push(part);
      renderedLength += part.value.length;
    } else {
      kept.push({ ...part, value: part.value.slice(0, remaining) });
      renderedLength += remaining;
      break;
    }
  }
  const truncated = renderedLength < parts.reduce((sum, p) => sum + p.value.length, 0);
  return { parts: kept, renderedLength, truncated };
}

function Panel({ children, frameSelector }) {
  return (
    <div className="rounded-lg border border-line bg-surface shadow-panel">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <FileSearch className="h-4 w-4 text-primary" aria-hidden="true" />
        <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted">Correction Ledger</h2>
        {frameSelector && <div className="ml-auto">{frameSelector}</div>}
      </div>
      {children}
    </div>
  );
}

function LegendSwatch({ colorClass, label }) {
  return (
    <span className="flex items-center gap-1.5 text-ink-muted">
      <span className={`font-mono font-semibold ${colorClass}`}>{label}</span>
    </span>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-16 text-center">
      <p className="text-sm text-ink-muted">Run a correction to see the annotated sequence here.</p>
      <p className="text-xs text-ink-faint">
        Matches are shown in nucleotide color; insertions and removals are highlighted inline.
      </p>
    </div>
  );
}
