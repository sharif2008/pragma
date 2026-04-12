# List all files to be reviewed

Produce a checklist of files that still need human review (per project `review.md`: code marked with `# REVIEW:`).

## What to do

1. Search the workspace for review markers in **source code**, not rule/command prose:
   - `// # REVIEW:`
   - `# # REVIEW:` (hash-style comments where used)
   - `/* # REVIEW:` or block comments that start a line with `# REVIEW:`
2. **Exclude** (do not list as “to review”):
   - `.cursor/rules/review.md`
   - `.cursor/commands/clean-review-tags.md`
   - `.cursor/commands/list-files-to-review.md`  
   when matches are only **documentation** describing the tag (e.g. “Use: # REVIEW:”).
3. Collect **unique file paths** that contain at least one real review tag in code (e.g. `**/*.{ts,tsx,js,jsx,py,sol,rs,go,...}` under project roots like `frontend/`, `hardhat-blockchain/`, etc.).
4. Output a **sorted list**:
   - One path per line (repo-relative).
   - If none: say clearly that no `# REVIEW:` tags were found in code.
5. Optionally append **per file**: line count of tags or first line number (if easy from search).

## Output format

Use a short markdown section:

```markdown
## Files to review

- `path/to/file.ts` (N tag(s))
```

Keep the list concise; no need to paste file contents unless the user asks.
