// Folder paths inside a dataset: "증강1/어둡게" ('' = the dataset's top level), joined with "/".

/** Natural order for folder and file names (증강2 before 증강10). */
export const byName = new Intl.Collator('ko', { numeric: true })

/** `child` inside `parent`; either may be '' (the top level). */
export function joinFolder(parent: string, child: string): string {
  return parent && child ? `${parent}/${child}` : parent || child
}

/** The folder that holds `path` ('' for a folder at the top level). */
export function parentOf(path: string): string {
  return path.slice(0, Math.max(0, path.lastIndexOf('/')))
}

/** `path` itself or a folder below it. */
export function within(path: string, folder: string): boolean {
  return path === folder || path.startsWith(folder + '/')
}

/** Where `path` is below `folder` ('' = in it). */
export function relativeTo(path: string, folder: string): string {
  return folder && within(path, folder) ? path.slice(folder.length + 1) : path
}
