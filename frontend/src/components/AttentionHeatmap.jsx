import { useEffect, useMemo, useRef, useState } from "react";
import { Activity } from "lucide-react";

const MAX_MATRIX_DIM = 220; // above this, block-average downsample rather than render 1:1 (see module notes)
const CELL_SIZE = 4; // rendered pixels per matrix cell

const LOW_COLOR = [18, 23, 31]; // surface color -- "no attention" fades into the panel
const HIGH_COLOR = [199, 125, 255]; // heat accent (Tailwind `heat`)

function lerpColor(t) {
  // Perceptual boost: attention distributions are typically peaky/sparse, so a
  // straight linear map leaves most of the image near-black with only a few
  // bright pixels. sqrt(t) keeps mid-low values visible without lying about
  // which cells actually received the most weight (order-preserving).
  const boosted = Math.sqrt(Math.max(0, Math.min(1, t)));
  const [r0, g0, b0] = LOW_COLOR;
  const [r1, g1, b1] = HIGH_COLOR;
  const r = Math.round(r0 + (r1 - r0) * boosted);
  const g = Math.round(g0 + (g1 - g0) * boosted);
  const b = Math.round(b0 + (b1 - b0) * boosted);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Real softmax attention over many source positions is almost always
 * tightly clustered well below 1.0 (verified empirically: an untrained
 * model's attention over a 48-token source spanned only ~0.018-0.028, not
 * anywhere near the theoretical 0-1 range) -- a WELL-trained, sharply
 * peaked attention distribution would still rarely put much mass near 1.0
 * across many positions. Mapping color against a fixed [0, 1] range would
 * crush that real variation into a single near-flat shade. Normalizing
 * against each matrix's own [min, max] instead makes the actual relative
 * differences visible regardless of the model's training state -- the
 * legend reports the true min/max being mapped so the scale is never
 * mistaken for an absolute one.
 */
function normalize(value, min, max) {
  if (max - min < 1e-12) return 0; // degenerate: every cell identical, avoid divide-by-zero
  return (value - min) / (max - min);
}

/** Block-average downsample: matrix (T x L) -> at most (maxRows x maxCols), each output cell = mean of its block. */
function downsample(matrix, maxRows, maxCols) {
  const rows = matrix.length;
  const cols = rows > 0 ? matrix[0].length : 0;
  if (rows <= maxRows && cols <= maxCols) {
    return { matrix, rowBlock: 1, colBlock: 1, downsampled: false };
  }
  const rowBlock = Math.ceil(rows / maxRows);
  const colBlock = Math.ceil(cols / maxCols);
  const outRows = Math.ceil(rows / rowBlock);
  const outCols = Math.ceil(cols / colBlock);

  const out = Array.from({ length: outRows }, () => new Array(outCols).fill(0));
  const counts = Array.from({ length: outRows }, () => new Array(outCols).fill(0));

  for (let t = 0; t < rows; t++) {
    const ot = Math.floor(t / rowBlock);
    for (let i = 0; i < cols; i++) {
      const oi = Math.floor(i / colBlock);
      out[ot][oi] += matrix[t][i];
      counts[ot][oi] += 1;
    }
  }
  for (let ot = 0; ot < outRows; ot++) {
    for (let oi = 0; oi < outCols; oi++) {
      if (counts[ot][oi] > 0) out[ot][oi] /= counts[ot][oi];
    }
  }
  return { matrix: out, rowBlock, colBlock, downsampled: true };
}

/**
 * AttentionHeatmap
 * -----------------
 * Renders one attention_chunk's (decode_steps x source_length) alignment
 * matrix as a canvas heatmap -- target/decode positions horizontal, source
 * positions vertical, per the design brief. Large matrices are block-average
 * downsampled (Part 4 spec: "downsampled to an optimal resolution") rather
 * than rendered 1:1 or silently cropped; the hover tooltip says so
 * explicitly when a cell represents an averaged block rather than a single
 * true attention coefficient, instead of presenting an average as if it
 * were exact.
 *
 * COLOR SCALE NOTE: mapped against each matrix's own [min, max], not a
 * fixed [0, 1] range. Verified empirically against a real API response:
 * softmax attention over dozens of source positions clusters tightly
 * (observed range ~0.018-0.028 for an untrained model over 48 positions) --
 * far from spanning 0-1 -- so a fixed-range map would render real,
 * meaningful variation as a single flat color. The legend reports the true
 * min/max being mapped, and hover always shows the raw, unnormalized
 * coefficient, so the adaptive scale is never mistaken for an absolute one.
 *
 * HONESTY NOTE on coordinates: source-axis positions map exactly to genome
 * coordinates (source_start + i, since chunking offsets are known exactly).
 * The target/decode axis is labeled as a decode step within this chunk, not
 * a corrected_sequence index -- the stitching step in
 * backend/inference_engine.py can trim a chunk's contribution, so a precise
 * decode-step-to-final-output-index mapping isn't available from the API
 * response as it stands. Labeling it as anything more precise would overstate
 * what's actually knowable here.
 */
export default function AttentionHeatmap({ attentionChunks }) {
  const [selectedChunkIndex, setSelectedChunkIndex] = useState(0);
  const [hoverCell, setHoverCell] = useState(null); // { t, i, value, blockAveraged }
  const canvasRef = useRef(null);

  const chunk = attentionChunks?.[selectedChunkIndex] ?? null;

  const { matrix, rowBlock, colBlock, downsampled } = useMemo(() => {
    if (!chunk) return { matrix: [], rowBlock: 1, colBlock: 1, downsampled: false };
    // chunk.attention_matrix is (decode_steps x source_length) = (T x L). We
    // display target(T) horizontally and source(L) vertically, so transpose
    // before downsampling/rendering.
    const T = chunk.attention_matrix.length;
    const L = T > 0 ? chunk.attention_matrix[0].length : 0;
    const transposed = Array.from({ length: L }, (_, i) => Array.from({ length: T }, (_, t) => chunk.attention_matrix[t][i]));
    return downsample(transposed, MAX_MATRIX_DIM, MAX_MATRIX_DIM);
  }, [chunk]);

  const { min: matrixMin, max: matrixMax } = useMemo(() => {
    if (matrix.length === 0) return { min: 0, max: 1 };
    let min = Infinity;
    let max = -Infinity;
    for (const row of matrix) {
      for (const value of row) {
        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
    return { min, max };
  }, [matrix]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || matrix.length === 0) return;
    const rows = matrix.length;
    const cols = matrix[0].length;
    canvas.width = cols * CELL_SIZE;
    canvas.height = rows * CELL_SIZE;

    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        ctx.fillStyle = lerpColor(normalize(matrix[row][col], matrixMin, matrixMax));
        ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
      }
    }
  }, [matrix, matrixMin, matrixMax]);

  const handleMouseMove = (event) => {
    if (matrix.length === 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.floor(x / CELL_SIZE);
    const row = Math.floor(y / CELL_SIZE);
    if (row < 0 || row >= matrix.length || col < 0 || col >= matrix[0].length) return;

    setHoverCell({
      sourcePosition: (chunk?.source_start ?? 0) + row * rowBlock,
      decodeStep: col * colBlock,
      value: matrix[row][col],
      blockAveraged: downsampled,
    });
  };

  if (!attentionChunks || attentionChunks.length === 0) {
    return (
      <PanelShell>
        <div className="flex flex-col items-center justify-center gap-2 px-4 py-16 text-center">
          <p className="text-sm text-ink-muted">Attention weights will appear here after a correction runs.</p>
          <p className="text-xs text-ink-faint">
            Shows which source positions the decoder focused on when reconstructing each region.
          </p>
        </div>
      </PanelShell>
    );
  }

  return (
    <PanelShell>
      {attentionChunks.length > 1 && (
        <div className="flex flex-wrap gap-1 border-b border-line px-4 py-2">
          {attentionChunks.map((c, idx) => (
            <button
              key={c.chunk_index}
              onClick={() => setSelectedChunkIndex(idx)}
              className={`rounded px-2 py-1 font-mono text-xs transition-colors duration-200 cursor-pointer ${
                idx === selectedChunkIndex
                  ? "bg-surface-raised text-ink-primary"
                  : "text-ink-muted hover:text-ink-primary"
              }`}
            >
              chunk {idx} · {c.source_start}–{c.source_end}
            </button>
          ))}
        </div>
      )}

      <div className="px-4 py-3">
        <div className="flex gap-3">
          <span
            className="shrink-0 self-stretch text-center font-mono text-[10px] uppercase tracking-widest text-ink-faint"
            style={{ writingMode: "vertical-rl" }}
          >
            source position →
          </span>
          <div className="overflow-auto rounded border border-line">
            <canvas
              ref={canvasRef}
              onMouseMove={handleMouseMove}
              onMouseLeave={() => setHoverCell(null)}
              className="block cursor-crosshair"
            />
          </div>
        </div>
        <p className="mt-1 pl-6 text-center font-mono text-[10px] uppercase tracking-widest text-ink-faint">
          decode step →
        </p>

        <div className="mt-3 flex items-center justify-between gap-4 font-mono text-xs">
          <div className="flex items-center gap-2 text-ink-muted">
            <span className="text-ink-faint">{matrixMin.toFixed(4)}</span>
            <span className="h-2 w-24 rounded-full" style={{ background: `linear-gradient(to right, ${lerpColor(0)}, ${lerpColor(1)})` }} />
            <span className="text-ink-faint">{matrixMax.toFixed(4)}</span>
          </div>

          <div className="min-w-[220px] text-right text-ink-muted">
            {hoverCell ? (
              <span>
                genome pos <span className="text-ink-primary">{hoverCell.sourcePosition}</span>, decode step{" "}
                <span className="text-ink-primary">{hoverCell.decodeStep}</span>: α ={" "}
                <span className="text-heat">{hoverCell.value.toFixed(4)}</span>
                {hoverCell.blockAveraged && <span className="text-ink-faint"> (block-averaged)</span>}
              </span>
            ) : (
              <span className="text-ink-faint">hover the grid for exact coefficients</span>
            )}
          </div>
        </div>
        {downsampled && (
          <p className="mt-1 text-xs text-ink-faint">
            Matrix downsampled for display (block averaging, {rowBlock}×{colBlock} cells per block) -- original
            resolution is {chunk.attention_matrix[0]?.length ?? 0} source × {chunk.attention_matrix.length} decode
            steps.
          </p>
        )}
        <p className="mt-1 text-xs text-ink-faint">
          Color scale is normalized to this chunk's own min/max, not an absolute 0-1 range -- softmax attention
          over many source positions is typically far tighter than that even when well-trained. Hover for the
          true, unnormalized coefficient.
        </p>
      </div>
    </PanelShell>
  );
}

function PanelShell({ children }) {
  return (
    <div className="rounded-lg border border-line bg-surface shadow-panel">
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <Activity className="h-4 w-4 text-heat" aria-hidden="true" />
        <h2 className="font-mono text-xs uppercase tracking-widest text-ink-muted">Attention Track</h2>
      </div>
      {children}
    </div>
  );
}
