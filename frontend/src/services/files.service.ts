import type { ManagedFileOut, FileUploadResponse } from 'src/api/types';

import { paths } from './paths';
import { requestJson, requestVoid, postMultipart } from './http-client';

export async function uploadFile(file: File, replacePublicId?: string | null): Promise<FileUploadResponse> {
  return postMultipart<FileUploadResponse>(paths.files.upload, file, {
    replace_public_id: replacePublicId ?? undefined,
  });
}

export async function listFiles(): Promise<ManagedFileOut[]> {
  return requestJson<ManagedFileOut[]>(paths.files.list);
}

export async function deleteFile(publicId: string): Promise<void> {
  return requestVoid(paths.files.byPublicId(publicId), { method: 'DELETE' });
}
