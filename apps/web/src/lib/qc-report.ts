/** Saved with each generated video so history does not mislabel unchecked media. */
export type QCStatus = "passed" | "failed" | "unavailable" | "not_checked" | "unknown";

export interface VideoQCReport {
  visual: QCStatus;
  audio: QCStatus;
  visualWarnings: string[];
  audioWarnings: string[];
  visualRetries: number;
  audioRetakes: number;
}

const VALID_STATUSES: readonly QCStatus[] = ["passed", "failed", "unavailable", "not_checked", "unknown"];

export function resolveQCStatus(
  status: QCStatus | null | undefined,
  passed: boolean | null | undefined,
  checkedByRequest: boolean,
  warnings: string[] = [],
): QCStatus {
  if (status && VALID_STATUSES.includes(status)) return status;
  // Backwards compatibility for video records made before explicit QC states.
  if (passed === false) return "failed";
  if (passed === true) return "passed";
  if (!checkedByRequest) return "not_checked";
  return warnings.length ? "unavailable" : "unknown";
}

export function aggregateQCStatus(statuses: QCStatus[]): QCStatus {
  if (!statuses.length) return "unknown";
  if (statuses.includes("failed")) return "failed";
  if (statuses.includes("unavailable")) return "unavailable";
  if (statuses.includes("unknown")) return "unknown";
  if (statuses.every((status) => status === "passed")) return "passed";
  if (statuses.every((status) => status === "not_checked")) return "not_checked";
  // A few passing scenes do not certify uninspected scenes.
  return "unknown";
}

export function qcStatusLabel(status: QCStatus): string {
  switch (status) {
    case "passed": return "Passed";
    case "failed": return "Failed — review needed";
    case "unavailable": return "Unavailable";
    case "not_checked": return "Not checked";
    default: return "Not recorded";
  }
}
