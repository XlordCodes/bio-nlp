import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";

const NUM_BUCKETS = 200;

/**
 * Buckets the rendered diff (same `parts` DiffViewer renders, so the
 * minimap's coordinate space matches what's actually on screen -- not the
 * pure corrected_sequence string, which would drift out of sync with
 * on-screen position the moment removed/added markers are rendered inline)
 * into NUM_BUCKETS segments, each scored by the fraction of edited
 * (added/removed) characters it contains.
 */
function computeDensityBuckets(parts, renderedLength, numBuckets) {
  if (renderedLength === 0) return new Array(numBuckets).fill(0);
  const bucketSize = renderedLength / numBuckets;
  const editCounts = new Array(numBuckets).fill(0);
  const totalCounts = new Array(numBuckets).fill(0);

  let pos = 0;
  for (const part of parts) {
    const isEdit = Boolean(part.added || part.removed);
    for (let i = 0; i < part.value.length; i++) {
      const bucket = Math.min(numBuckets - 1, Math.floor(pos / bucketSize));
      totalCounts[bucket] += 1;
      if (isEdit) editCounts[bucket] += 1;
      pos += 1;
    }
  }
  return editCounts.map((count, i) => (totalCounts[i] > 0 ? count / totalCounts[i] : 0));
}

/**
 * Minimap
 * --------
 * A slim horizontal density strip sitting above the Ledger, bucketing the
 * full (possibly truncated-for-display) sequence into ~200 segments
 * colored by local edit density -- lets a user spot and jump to the
 * regions with the most corrections instead of scrolling blindly through
 * a long read. The viewport rectangle reflects the Ledger's real scroll
 * position (kept in sync via a scroll listener + ResizeObserver) and can
 * be dragged, or the track clicked directly, to scroll the Ledger.
 */
export default function Minimap({ parts, renderedLength, scrollContainerRef }) {
  const trackRef = useRef(null);
  const [viewport, setViewport] = useState({ left: 0, width: 1 });
  const [isDragging, setIsDragging] = useState(false);

  const densities = useMemo(
    () => computeDensityBuckets(parts, renderedLength, NUM_BUCKETS),
    [parts, renderedLength]
  );

  const readViewportFromScroll = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container || container.scrollWidth <= 0) return;
    const { scrollLeft, scrollWidth, clientWidth } = container;
    setViewport({
      left: scrollLeft / scrollWidth,
      width: Math.min(1, clientWidth / scrollWidth),
    });
  }, [scrollContainerRef]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    readViewportFromScroll();
    container.addEventListener("scroll", readViewportFromScroll, { passive: true });
    const resizeObserver = new ResizeObserver(readViewportFromScroll);
    resizeObserver.observe(container);
    return () => {
      container.removeEventListener("scroll", readViewportFromScroll);
      resizeObserver.disconnect();
    };
  }, [scrollContainerRef, readViewportFromScroll, renderedLength]);

  const scrollToFraction = useCallback(
    (fraction, centerOnViewport = true) => {
      const container = scrollContainerRef.current;
      if (!container) return;
      const { scrollWidth, clientWidth } = container;
      const target = centerOnViewport
        ? fraction * scrollWidth - clientWidth / 2
        : fraction * scrollWidth;
      const maxScroll = Math.max(0, scrollWidth - clientWidth);
      container.scrollLeft = Math.max(0, Math.min(maxScroll, target));
    },
    [scrollContainerRef]
  );

  const fractionFromClientX = useCallback((clientX) => {
    const track = trackRef.current;
    if (!track) return 0;
    const rect = track.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }, []);

  const handleTrackClick = (event) => {
    if (isDragging) return;
    scrollToFraction(fractionFromClientX(event.clientX));
  };

  const handleViewportPointerDown = (event) => {
    event.stopPropagation();
    setIsDragging(true);
    const track = trackRef.current;
    const startClientX = event.clientX;
    const startLeft = viewport.left;

    const handlePointerMove = (moveEvent) => {
      const track_rect = track.getBoundingClientRect();
      const deltaFraction = (moveEvent.clientX - startClientX) / track_rect.width;
      scrollToFraction(startLeft + deltaFraction, false);
    };
    const handlePointerUp = () => {
      setIsDragging(false);
      document.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("pointerup", handlePointerUp);
    };
    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);
  };

  if (renderedLength === 0) return null;

  return (
    <div className="px-4 pt-3">
      <div
        ref={trackRef}
        onClick={handleTrackClick}
        className="relative flex h-6 w-full cursor-pointer overflow-hidden rounded-sm border border-line bg-void"
        role="scrollbar"
        aria-label="Sequence overview -- click or drag to navigate"
        aria-orientation="horizontal"
        aria-valuenow={Math.round(viewport.left * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {densities.map((density, i) => (
          <div
            key={i}
            className="h-full flex-1"
            style={{ backgroundColor: `rgba(199, 125, 255, ${0.08 + density * 0.75})` }}
          />
        ))}

        <motion.div
          onPointerDown={handleViewportPointerDown}
          whileHover={{ opacity: 0.9 }}
          className="absolute top-0 h-full cursor-grab border-x-2 border-primary bg-primary/15 active:cursor-grabbing"
          style={{
            left: `${viewport.left * 100}%`,
            width: `${Math.max(viewport.width * 100, 2)}%`,
          }}
        />
      </div>
      <p className="mt-1 text-[10px] uppercase tracking-widest text-ink-faint">
        overview -- density of corrections across the full read
      </p>
    </div>
  );
}
