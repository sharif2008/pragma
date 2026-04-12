# Clean all `# REVIEW:` tags

Use this when the user wants review markers stripped after a change is finalized.

## What to do

1. Search the repository for lines containing `# REVIEW:` (including `// # REVIEW:` and similar single-line comment forms).
2. Remove **only** those review-tag lines. Do not delete unrelated comments or code.
3. If a `# REVIEW:` line was the only content on its line next to real code, leave the code unchanged.
4. Keep normal documentation comments where they still add value; drop redundant wording left empty by removing the tag.
5. Run lint / `tsc --noEmit` (or the project’s usual check) on touched packages when TypeScript or JavaScript files changed.

## Do not

- Strip `REVIEW` when it appears inside non-tag prose (e.g. variable names like `codeReview`) unless it is clearly the same tag convention.
- Remove `review.md` or rule files; this task targets **code comment tags** only.
