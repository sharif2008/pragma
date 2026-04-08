export type TimeSortOrder = 'asc' | 'desc';

export function timeValueMs(v: string | number | Date | null | undefined): number {
  if (v == null || v === '') return 0;
  if (typeof v === 'number') return Number.isFinite(v) ? v : 0;
  if (v instanceof Date) return v.getTime();
  const t = Date.parse(String(v));
  return Number.isNaN(t) ? 0 : t;
}

/** Stable copy sorted by a timestamp field. `desc` = newest first. */
export function sortByTime<T>(
  rows: readonly T[],
  getTime: (row: T) => string | number | Date | null | undefined,
  order: TimeSortOrder
): T[] {
  const dir = order === 'desc' ? -1 : 1;
  return [...rows].sort((a, b) => {
    const ta = timeValueMs(getTime(a));
    const tb = timeValueMs(getTime(b));
    if (ta === tb) return 0;
    return ta < tb ? -dir : dir;
  });
}

export function toggleTimeSortOrder(o: TimeSortOrder): TimeSortOrder {
  return o === 'desc' ? 'asc' : 'desc';
}
