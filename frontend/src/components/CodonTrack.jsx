import { useMemo } from "react";

// Scoped deliberately to STOP-codon detection only, not full amino-acid
// translation: this is a small, verifiable set (the universal genetic
// code's 3 stop codons), whereas the full 64-entry codon table is easy to
// get subtly wrong from memory in a way that's hard to self-check quickly.
// STOP detection is also the actual promised feature -- it's the direct
// biological signal a frameshift error produces (an uncorrected indel
// shifts the frame and can introduce a premature stop, truncating the
// protein), which is what this track exists to surface.
const STOP_CODONS = new Set(["TAA", "TAG", "TGA"]);

/**
 * Assigns each REAL (non-removed) character in the diff to a codon index
 * and position-within-codon, based on the chosen reading frame offset --
 * "real" specifically excludes characters from `removed` diff parts, since
 * those don't exist in the corrected sequence and have no place in its
 * reading frame at all. Returns one cell per rendered character, in the
 * exact same order/count as DiffViewer's own flattened rendering, so the
 * two rows stay pixel-aligned in the shared scroll container.
 */
function computeCodonCells(parts, frameOffset) {
  const cells = [];
  const realCellRefs = [];

  for (const part of parts) {
    for (const char of part.value) {
      if (part.removed) {
        cells.push({ gap: true });
      } else {
        const cell = { gap: false, char };
        cells.push(cell);
        realCellRefs.push(cell);
      }
    }
  }

  const codonGroups = new Map();
  for (let i = 0; i < realCellRefs.length; i++) {
    const adjusted = i - frameOffset;
    if (adjusted < 0) continue; // before the first full codon in this frame
    const codonIndex = Math.floor(adjusted / 3);
    const posInCodon = adjusted % 3;
    realCellRefs[i].codonIndex = codonIndex;
    realCellRefs[i].posInCodon = posInCodon;
    if (!codonGroups.has(codonIndex)) codonGroups.set(codonIndex, []);
    codonGroups.get(codonIndex).push(realCellRefs[i]);
  }

  for (const refs of codonGroups.values()) {
    if (refs.length === 3) {
      const codon = refs.map((r) => r.char.toUpperCase()).join("");
      const isStop = STOP_CODONS.has(codon);
      refs.forEach((r) => {
        r.isStop = isStop;
      });
    }
  }

  return cells;
}

/**
 * CodonTrack
 * -----------
 * Renders directly beneath the Ledger's sequence row, in the SAME scroll
 * container (not a separately JS-synced scroll region -- simpler, and
 * avoids the jank/lag a two-container synced-scroll approach can introduce)
 * so it stays trivially, perfectly aligned with the ledger above it.
 * Alternates a subtle background tint every 3 real bases to mark codon
 * boundaries in the chosen reading frame, and flags premature stop codons.
 */
export default function CodonTrack({ parts, frameOffset }) {
  const cells = useMemo(() => computeCodonCells(parts, frameOffset), [parts, frameOffset]);
  const stopCount = useMemo(() => cells.filter((c) => c.isStop).length, [cells]);

  return (
    <div>
      <div className="min-w-max whitespace-pre font-mono text-sm leading-relaxed">
        {cells.map((cell, i) => {
          if (cell.gap) {
            return <span key={i}> </span>;
          }
          if (cell.isStop) {
            return (
              <span key={i} className="bg-destructive/40 text-destructive" title="Premature stop codon">
                {cell.char}
              </span>
            );
          }
          const shaded = cell.codonIndex !== undefined && cell.codonIndex % 2 === 0;
          return (
            <span key={i} className={shaded ? "bg-surface-raised text-ink-faint" : "text-ink-faint"}>
              {cell.char}
            </span>
          );
        })}
      </div>
      {stopCount > 0 && (
        <p className="mt-1 text-xs text-destructive">
          {stopCount} premature stop codon{stopCount !== 1 ? "s" : ""} in this frame -- likely an uncorrected
          frameshift.
        </p>
      )}
    </div>
  );
}

export function FrameSelector({ frameOffset, onChange }) {
  return (
    <div className="flex items-center gap-1 rounded-md bg-void p-1">
      {[0, 1, 2].map((offset) => (
        <button
          key={offset}
          onClick={() => onChange(offset)}
          className={`rounded px-2 py-1 font-mono text-xs transition-colors duration-200 cursor-pointer ${
            frameOffset === offset ? "bg-surface-raised text-ink-primary" : "text-ink-muted hover:text-ink-primary"
          }`}
        >
          frame {offset + 1}
        </button>
      ))}
    </div>
  );
}
