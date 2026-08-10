/**
 * api/client.js
 * ---------------
 * Thin wrapper around the FastAPI backend (backend/main.py). Centralizes
 * the three real endpoints -- /health, /correct, /correct/file -- so
 * components never construct fetch() calls or parse error shapes
 * themselves. Matches backend/schemas.py's InferenceResponse /
 * ErrorResponse contracts exactly.
 *
 * Base URL: in dev, Vite's proxy (vite.config.js) forwards /api/* to
 * http://localhost:8000, so no CORS setup is needed locally. In
 * production, set VITE_API_BASE_URL to the deployed backend's real
 * origin (e.g. https://api.example.com) -- see .env.example.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function parseErrorResponse(response) {
  try {
    const body = await response.json();
    return body?.detail ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

/** GET /health -> { status, model_loaded, device } */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError("Health check failed", response.status, detail);
  }
  return response.json();
}

/**
 * POST /correct with a raw sequence string.
 * Returns InferenceResponse: { corrected_sequence, metrics, attention_chunks }
 */
export async function correctSequence(sequence) {
  const response = await fetch(`${API_BASE_URL}/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError("Correction request failed", response.status, detail);
  }
  return response.json();
}

/**
 * POST /correct/file with a single-record .fasta/.fa File object.
 * Returns the same InferenceResponse shape as correctSequence().
 */
export async function correctSequenceFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/correct/file`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const detail = await parseErrorResponse(response);
    throw new ApiError("File correction request failed", response.status, detail);
  }
  return response.json();
}

export { ApiError };
