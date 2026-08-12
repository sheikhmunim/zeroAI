# Git Workflow — Branches, PRs, and How Not to Lose Work

A practical guide to the branch-per-change workflow, with every command you
actually need. Written for this repo, but the workflow is the standard one used
almost everywhere.

> **Golden rule:** `main` is always deployable. You never work directly on it.
> In this repo that is literal — a push to `main` deploys to production.

---

# Contents

- [1. The mental model](#1-the-mental-model)
- [2. The daily loop](#2-the-daily-loop)
- [3. Branches](#3-branches)
- [4. Commits](#4-commits)
- [5. Pull requests](#5-pull-requests)
- [6. Keeping a branch current](#6-keeping-a-branch-current)
- [7. Merge conflicts](#7-merge-conflicts)
- [8. Undoing things — the "oh no" section](#8-undoing-things--the-oh-no-section)
- [9. Inspecting state](#9-inspecting-state)
- [10. Branch protection](#10-branch-protection)
- [11. How this interacts with CI/CD here](#11-how-this-interacts-with-cicd-here)
- [12. Dangerous commands](#12-dangerous-commands)
- [13. Quick reference](#13-quick-reference)

---

# 1. The mental model

A branch is **not a copy of your files**. Git stores it as a pointer to a
commit, so creating one is instant and costs a few bytes.

```
main       A───B───C                    ← always deployable, always green
                    \
add-tests            D───E              ← your work, isolated and breakable
                          \
fix-cors                   ...          ← someone else's, in parallel
```

You can be completely broken on your branch. `main` doesn't care. That
isolation is the entire point.

**Three states a file can be in:**

```
working directory  →  staging area  →  committed
     (edited)          (git add)       (git commit)
```

`git status` tells you where everything is. When confused, run it.

---

# 2. The daily loop

95% of your git usage is these eight commands:

```bash
git switch main                       # go to main
git pull                              # get everyone's latest

git switch -c add-pipeline-tests      # create + switch to a new branch

# ...edit files...

git status                            # what changed?
git diff                              # what exactly changed?
git add .                             # stage everything
git commit -m "Add unit tests for stratified_split"
git push -u origin add-pipeline-tests # publish the branch
```

Then open a PR (§5), get it green, merge.

> `git switch` is the modern command for changing branches; `git checkout` still
> works and is what older tutorials use. `switch` exists because `checkout` did
> too many unrelated things.

---

# 3. Branches

### Create

```bash
git switch -c my-feature              # branch from where you are
git switch -c my-feature main         # branch explicitly from main
git switch -c my-feature origin/main  # branch from remote main
```

### Move between

```bash
git switch main
git switch -                          # back to the previous branch
```

### List

```bash
git branch                            # local
git branch -r                         # remote
git branch -a                         # all
git branch -vv                        # with tracking info + last commit
```

### Rename

```bash
git branch -m new-name                # rename current branch
git branch -m old-name new-name
```

### Delete

```bash
git branch -d my-feature              # safe: refuses if unmerged
git branch -D my-feature              # force, even if unmerged
git push origin --delete my-feature   # delete the remote copy
```

### Naming

Use a short, descriptive, hyphenated name. Common prefixes:

```
add-pipeline-tests
fix-cors-on-staging
bump-torch-2.14
docs-deployment-runbook
```

Some teams use `feat/`, `fix/`, `chore/` prefixes. Follow whatever your team
does; consistency matters more than the scheme.

### Sizing — the thing beginners get wrong

**One branch = one thing you could describe in a single sentence.** Hours to a
couple of days, not weeks.

Small branches merge cleanly. A three-week branch becomes archaeology: it
conflicts with everything, nobody wants to review 4,000 changed lines, and you
can't ship any of it until all of it works.

If you're doing two unrelated things on one branch, split it.

---

# 4. Commits

### Stage and commit

```bash
git add .                             # everything
git add src/api.py                    # one file
git add src/                          # a directory
git add -p                            # interactively, hunk by hunk

git commit -m "Short summary in the imperative mood"
git commit                            # opens an editor for a longer message
git commit -am "msg"                  # add + commit tracked files (skips new ones)
```

### Unstage / discard

```bash
git restore --staged src/api.py       # unstage, keep the edit
git restore src/api.py                # DISCARD the edit (unrecoverable)
git restore .                         # discard ALL uncommitted edits
```

### Amend the last commit

```bash
git commit --amend -m "Better message"
git add forgotten-file.py && git commit --amend --no-edit
```

> ⚠️ Amending **rewrites history**. Fine before pushing. After pushing you need
> `git push --force-with-lease`, which is only acceptable on your own branch —
> never on a shared one.

### Message style

```
Add unit tests for the data pipeline

stratified_split and sample_balanced_indices had no coverage. A silent
inversion of the CIFAKE label map would produce ~5% accuracy with no
exception anywhere, so these assert the mapping explicitly.
```

- First line: imperative ("Add", not "Added"), under ~72 characters, no full stop
- Blank line
- Body: **why**, not what. The diff already shows what.

Commit often on your branch. Squash on merge if you want a clean history.

---

# 5. Pull requests

A PR is a request to merge your branch into `main`, plus the place review and
CI happen. **It is the review unit** — you cannot review a change already in
`main`.

### Via the website

1. `git push -u origin my-branch`
2. GitHub shows a "Compare & pull request" banner — click it
3. Or go to https://github.com/sheikhmunim/zeroAI/pulls → **New pull request**
4. base: `main` ← compare: `my-branch`
5. Title + description, **Create pull request**

CI starts immediately. In this repo that means `lint`, `test` and `smoke` run —
and **`deploy` does not**, because it requires `event_name == 'push'`.

### Via the `gh` CLI (optional, faster)

Install: `winget install GitHub.cli`, then `gh auth login`.

```bash
gh pr create --fill                   # title/body from your commits
gh pr create --title "Add tests" --body "Covers stratified_split"
gh pr create --draft                  # not ready for review yet

gh pr status                          # your PRs
gh pr list
gh pr view --web                      # open in browser
gh pr checks                          # CI status
gh pr diff
```

### Merging

```bash
gh pr merge --squash --delete-branch     # recommended
gh pr merge --merge                      # keep every commit
gh pr merge --rebase                     # replay commits onto main
```

Or press the button on the website.

| strategy | result | when |
|---|---|---|
| **Squash** | all commits become one on `main` | default; keeps history readable |
| Merge | keeps every commit + a merge commit | when individual commits matter |
| Rebase | replays commits, no merge commit | linear history, no merge commits |

**Squash is the sane default** — your seven "wip" commits become one clean entry.

---

# 6. Keeping a branch current

While you work, `main` moves. Pull those changes in regularly so the gap stays
small.

### Merge (simple, safe)

```bash
git switch my-feature
git fetch origin
git merge origin/main
```

Creates a merge commit. History shows exactly what happened. **Use this if
you're unsure.**

### Rebase (clean, rewrites history)

```bash
git switch my-feature
git fetch origin
git rebase origin/main
```

Replays your commits on top of the latest `main`, as if you'd branched just
now. Linear, tidy history.

> ⚠️ Rebasing rewrites your commits, so a pushed branch then needs
> `git push --force-with-lease`. **Never rebase a branch other people are
> working on.**

### Rule of thumb

**Rebase your own unshared branch. Merge everything else.**

---

# 7. Merge conflicts

Git stops when two changes touch the same lines and it can't decide.

```
CONFLICT (content): Merge conflict in src/api.py
Automatic merge failed; fix conflicts and then commit the result.
```

### What you'll see in the file

```
<<<<<<< HEAD
threshold: float = Query(0.5, ge=0.0, le=1.0),
=======
threshold: float = Query(0.5, ge=0.01, le=0.99),
>>>>>>> origin/main
```

- Above `=======` — **your** version
- Below — **theirs**

### Resolving

```bash
git status                            # lists conflicted files
# edit each file: pick one side, or combine, then DELETE the <<<< ==== >>>> markers
git add src/api.py                    # marks it resolved
git commit                            # (merge) — or:
git rebase --continue                 # (rebase)
```

### Bail out

```bash
git merge --abort
git rebase --abort
```

Puts you back exactly where you started. Safe.

### Take one side wholesale

```bash
git checkout --ours src/api.py        # keep your version
git checkout --theirs src/api.py      # keep theirs
git add src/api.py
```

**Prevention beats resolution:** short-lived branches, merged often.

---

# 8. Undoing things — the "oh no" section

Ordered by what you're trying to undo.

### I edited a file and want the last committed version back

```bash
git restore src/api.py                # ⚠️ unrecoverable
git restore .                         # all files
```

### I staged something by mistake

```bash
git restore --staged src/api.py       # unstage, edit is preserved
```

### I want to change the last commit message

```bash
git commit --amend -m "Correct message"
```

### I committed but haven't pushed — undo it

```bash
git reset --soft HEAD~1               # undo commit, KEEP changes staged
git reset HEAD~1                      # undo commit, keep changes unstaged
git reset --hard HEAD~1               # ⚠️ undo commit AND DELETE the changes
```

`--soft` is almost always what you want.

### I already pushed and need to undo it

```bash
git revert <commit-sha>               # makes a NEW commit that undoes it
git push
```

**Use `revert`, not `reset`, for anything already pushed.** Revert adds history
rather than rewriting it, so it doesn't break anyone else's clone.

### I need to switch branches but I'm mid-change

```bash
git stash                             # shelve everything
git stash -u                          # include untracked files
git switch other-branch
# ...
git switch -
git stash pop                         # bring it back

git stash list
git stash drop
git stash clear                       # ⚠️ deletes all stashes
```

### I committed to `main` by accident

```bash
git switch -c my-feature              # take the commits onto a new branch
git switch main
git reset --hard origin/main          # ⚠️ reset main to the remote
git switch my-feature
```

### I think I destroyed work

```bash
git reflog                            # every HEAD position for ~90 days
git switch -c rescue <sha-from-reflog>
```

**`git reflog` recovers almost anything that was ever committed.** If you have
committed at some point, it is very likely still recoverable. Try this before
panicking.

### I want to grab one commit from another branch

```bash
git cherry-pick <commit-sha>
```

---

# 9. Inspecting state

```bash
git status                            # the one you'll run most
git status -sb                        # compact

git log --oneline -10
git log --oneline --graph --all       # visual branch structure
git log -p src/api.py                 # history of one file, with diffs
git log --author="Munim"
git log main..my-feature              # commits on my branch, not on main

git diff                              # unstaged changes
git diff --staged                     # staged changes
git diff main                         # my branch vs main
git diff HEAD~1                       # vs the previous commit
git diff --stat                       # summary only

git show <sha>                        # one commit in full
git blame src/api.py                  # who last touched each line

git remote -v
git branch -vv
```

---

# 10. Branch protection

Most companies configure `main` so direct pushes are **rejected** — the only
way in is a PR with passing CI and an approval.

Set it up on your own repo:

1. https://github.com/sheikhmunim/zeroAI/settings/branches
2. **Add branch ruleset** (or classic **Add rule**), branch name `main`
3. Enable:
   - ☑ Require a pull request before merging
   - ☑ Require status checks to pass — select `lint`, `test`, `smoke`
   - ☑ Require branches to be up to date before merging
   - ☑ Block force pushes

With that on, `git push origin main` fails with a clear error, and the workflow
in §2 becomes the only path. **This is worth turning on even solo** — it makes
accidentally shipping something impossible rather than merely unlikely.

---

# 11. How this interacts with CI/CD here

| action | `lint` `test` `smoke` | `deploy` |
|---|---|---|
| push to a feature branch | ❌ not triggered | ❌ |
| open / update a PR into `main` | ✅ runs | ❌ blocked by `if:` |
| merge the PR into `main` | ✅ runs | ✅ **deploys to production** |
| push directly to `main` | ✅ runs | ✅ **deploys to production** |

`deploy` requires `github.event_name == 'push' && github.ref == 'refs/heads/main'`,
so a pull request — including one from a fork — can never ship code.

### Run CI on every branch too

Feature-branch pushes currently trigger nothing. To get feedback earlier, edit
`.github/workflows/ci.yml`:

```yaml
on:
  push:
    branches: ['**']        # every branch
  pull_request:
    branches: [main]
```

`deploy` still won't fire — it checks the ref.

### Skip CI for docs-only changes

```yaml
on:
  push:
    branches: [main]
    paths-ignore: ['**.md', 'docs/**']
```

Rebuilding a 2.35 GB image because you fixed a typo is waste.

### Skip CI for one commit

```bash
git commit -m "Fix typo [skip ci]"
```

---

# 12. Dangerous commands

Know these so you recognise them, and think before running them.

| command | what it destroys |
|---|---|
| `git reset --hard` | all uncommitted work, unrecoverably |
| `git restore <file>` | that file's uncommitted edits |
| `git clean -fd` | all untracked files and directories |
| `git push --force` | **other people's commits on the remote** |
| `git branch -D` | an unmerged branch (recoverable via reflog) |
| `git stash clear` | every stash |

**If you must force-push, use `--force-with-lease`:**

```bash
git push --force-with-lease
```

It refuses if the remote has commits you haven't seen — so you can't silently
destroy someone else's work. Plain `--force` will happily do exactly that.

---

# 13. Quick reference

```bash
# --- the loop -------------------------------------------------------------
git switch main && git pull
git switch -c my-feature
git add . && git commit -m "message"
git push -u origin my-feature
gh pr create --fill                # or open it on the website
gh pr merge --squash --delete-branch

# --- see what's going on --------------------------------------------------
git status
git diff
git log --oneline --graph --all
git branch -vv

# --- keep current ---------------------------------------------------------
git fetch origin
git merge origin/main              # safe
git rebase origin/main             # clean, rewrites history

# --- undo -----------------------------------------------------------------
git restore --staged <file>        # unstage
git restore <file>                 # discard edit
git commit --amend                 # fix last commit
git reset --soft HEAD~1            # undo commit, keep changes
git revert <sha>                   # undo a PUSHED commit
git stash / git stash pop          # shelve work
git reflog                         # recover almost anything

# --- conflicts ------------------------------------------------------------
git status                         # what conflicted
# edit, remove <<<< ==== >>>> markers
git add <file>
git commit                         # or: git rebase --continue
git merge --abort                  # give up safely
```

---

## The habits that matter

1. **Never work on `main`.** Branch first, always.
2. **Small branches, merged often.** Hours or days, not weeks.
3. **`git status` when confused.** It is almost always the answer.
4. **`git reflog` before panicking.** Committed work is rarely truly lost.
5. **`revert` for pushed commits, `reset` for local ones.**
6. **`--force-with-lease`, never bare `--force`.**
7. **Commit messages explain *why*.** The diff already shows *what*.
