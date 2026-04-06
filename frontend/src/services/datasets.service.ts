import type { ManagedFileOut, DatasetPreviewOut, FileUploadResponse } from 'src/api/types';

import { paths } from './paths';
import { requestJson, requestVoid, postMultipart } from './http-client';

export async function uploadDataset(file: File, replacePublicId?: string | null): Promise<FileUploadResponse> {
  return postMultipart<FileUploadResponse>(paths.datasets.upload, file, {
    replace_public_id: replacePublicId ?? undefined,
  });
}

export async function listDatasets(): Promise<ManagedFileOut[]> {
  return requestJson<ManagedFileOut[]>(paths.datasets.list);
}

export async function getDatasetPreview(publicId: string, limit = 50): Promise<DatasetPreviewOut> {
  const q = new URLSearchParams({ limit: String(limit) });
  return requestJson<DatasetPreviewOut>(`${paths.datasets.preview(publicId)}?${q}`);
}

export async function deleteDataset(publicId: string): Promise<void> {
  return requestVoid(paths.datasets.byPublicId(publicId), { method: 'DELETE' });
}
