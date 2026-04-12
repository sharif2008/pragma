When modifying existing code:
- Add a short comment explaining what was changed and why.

When creating new classes or functions:
- Always include a clear comment describing purpose, inputs, and outputs.

For major changes:
- Add detailed comments explaining the logic and impact.

Add a review tag to all AI-generated or modified code:
- Use: # REVIEW:
- Place it above each changed or new block.

Ensure every important change has a corresponding # REVIEW: comment.

Keep comments concise and meaningful.

When requested, clean all # REVIEW: tags and keep only necessary documentation comments.

## Cursor shortcuts (type `/` in Chat or Agent)

| Command | File |
| -------- | ---- |
| **clean-review-tags** | `.cursor/commands/clean-review-tags.md` |
| **list-files-to-review** | `.cursor/commands/list-files-to-review.md` |
| **review-commit-push** | `.cursor/commands/review-commit-push.md` |