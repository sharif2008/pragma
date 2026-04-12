# Review changes, commit, and push

End-to-end: inspect the working tree, summarize what changed, commit with a solid message, and push.

## Before you start

1. Run `git status` from the repo root (`ChainAgentVFL`). Note branch name and whether there is anything to commit.
2. If there are no staged/unstaged changes worth committing, say so and stop (do not create an empty commit).

## Review

1. Show `git diff` for unstaged changes and `git diff --staged` if anything is already staged.
2. Give a **short review**: what changed, why it matters, any risks or follow-ups (tests not run, etc.).
3. If the project has quick checks (e.g. `frontend`: `npm run lint` / `npx tsc --noEmit`), run them when edits are in scope and report failures before committing.

## Commit

1. Stage with `git add` only what belongs in this commit (ask if unrelated files appear).
2. Write a **single conventional-style message**: imperative mood, complete first line (~72 chars), optional body for context.
   - Example: `Add Forest theme preset and slash commands for review workflow`
3. Run `git commit -m "..."` (use a heredoc or `-m` twice for body if needed on the user’s shell).

## Push

1. Run `git push` (set upstream with `git push -u origin <branch>` if the branch has no upstream yet).
2. Report success or paste the remote error and suggest a fix (auth, diverged branch, etc.).

## Safety

- Do not force-push (`--force`) unless the user explicitly asks.
- Do not skip hooks (`--no-verify`) unless the user explicitly asks.
- If multiple remotes exist, prefer `origin` and state which remote/branch was used.
