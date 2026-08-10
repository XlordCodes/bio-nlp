import { Toaster as Sonner } from "sonner";

/**
 * Themed wrapper around the `sonner` toast library, matching shadcn/ui's
 * standard integration pattern. Used for confirmation feedback on export
 * actions (download/copy) -- a less intrusive alternative to a persistent
 * inline banner for actions that succeed instantly.
 */
function Toaster(props) {
  return (
    <Sonner
      theme="dark"
      className="toaster group"
      position="bottom-right"
      toastOptions={{
        classNames: {
          toast:
            "group toast bg-popover text-ink-primary border border-line shadow-panel font-sans rounded-md",
          description: "text-ink-muted",
          actionButton: "bg-primary text-primary-foreground",
          cancelButton: "bg-surface-raised text-ink-muted",
        },
      }}
      {...props}
    />
  );
}

export { Toaster };
