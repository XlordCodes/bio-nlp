import { Download, Copy, FileJson, FileText } from "lucide-react";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "./ui/dropdown-menu.jsx";
import { Button } from "./ui/button.jsx";

function triggerDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function toFastaRecord(sequence, header) {
  const wrapped = sequence.match(/.{1,70}/g)?.join("\n") ?? sequence;
  return `>${header}\n${wrapped}\n`;
}

/**
 * ExportMenu
 * -----------
 * Surfaces the three ways to get a correction result out of the browser --
 * previously the only option was reading the ledger visually. All three
 * write client-side (Blob + object URL) rather than round-tripping through
 * the backend, since the full result is already in memory.
 */
export default function ExportMenu({ result }) {
  if (!result) return null;

  const handleDownloadFasta = () => {
    const fasta = toFastaRecord(result.corrected_sequence, "corrected_sequence");
    triggerDownload(fasta, "corrected_sequence.fasta", "text/plain;charset=utf-8");
    toast.success("Downloaded corrected_sequence.fasta");
  };

  const handleDownloadJson = () => {
    triggerDownload(JSON.stringify(result, null, 2), "correction_report.json", "application/json;charset=utf-8");
    toast.success("Downloaded correction_report.json");
  };

  const handleCopySequence = async () => {
    try {
      await navigator.clipboard.writeText(result.corrected_sequence);
      toast.success("Corrected sequence copied to clipboard");
    } catch {
      toast.error("Clipboard access was blocked by the browser");
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <Download className="h-3.5 w-3.5" aria-hidden="true" />
          Export
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuLabel>Export result</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={handleDownloadFasta}>
          <FileText className="h-4 w-4 text-ink-muted" aria-hidden="true" />
          Download FASTA
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={handleDownloadJson}>
          <FileJson className="h-4 w-4 text-ink-muted" aria-hidden="true" />
          Download JSON report
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={handleCopySequence}>
          <Copy className="h-4 w-4 text-ink-muted" aria-hidden="true" />
          Copy sequence
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
