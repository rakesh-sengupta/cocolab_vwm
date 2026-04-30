# GitHub setup walkthrough

This guide walks through getting `cocolab_vwm` into a properly structured
GitHub repository with branch protection, CI, and a sensible release flow.

## One-time setup (you only do this once)

### 1. Create the repo on GitHub

Go to <https://github.com/new> and create an empty repository named
`cocolab_vwm` under your account or the CoCoLab org. **Do not** initialise
with README, .gitignore, or license — we already have those locally.

If you want a CoCoLab-branded organisation account, create one first
(<https://github.com/organizations/new>) and create the repo under that.

### 2. Initialise the local repository

In the unpacked `cocolab_vwm/` directory:

```bash
git init
git add .
git status                 # sanity-check what's about to be committed
git commit -m "v0.2.0: OCOS hierarchy + pooling + BG-thalamus gate"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/cocolab_vwm.git
git push -u origin main
```

If you don't have SSH keys set up for GitHub, use HTTPS:

```bash
git remote add origin https://github.com/YOUR_USERNAME/cocolab_vwm.git
```

GitHub will prompt for a personal access token instead of a password —
generate one at <https://github.com/settings/tokens> with `repo` scope.

### 3. Create the `develop` branch

`main` is for tagged releases only. Day-to-day work happens on `develop`
and feature branches.

```bash
git checkout -b develop
git push -u origin develop
```

### 4. Tag the v0.2 release

```bash
git checkout main
git tag -a v0.2.0 -m "v0.2.0: pooling + BG gate"
git push --tags
```

On GitHub, go to the repo's Releases page and turn the v0.2.0 tag into a
proper release with the CHANGELOG.md v0.2.0 section as the description.

### 5. (Optional but strongly recommended) Branch protection rules

In the GitHub repo settings -> Branches, add a rule for `main`:

- Require a pull request before merging.
- Require status checks to pass before merging (select the `tests` workflow).
- Restrict who can push to matching branches (only you, for now).

This ensures `main` never has broken code on it.

### 6. Verify CI runs

Push any small change to `develop` and check the Actions tab. The
`tests` workflow should run on Python 3.10, 3.11, and 3.12, and pass.
If something fails (typically because of a version pin), update
`pyproject.toml` and push again.

## Day-to-day workflow

### Adding a new feature

```bash
git checkout develop
git pull
git checkout -b feature/my-new-thing

# Hack hack hack. Add tests. Run them.
pytest                               # fast tier
pytest --slow                        # full suite, before pushing
ruff check cocolab_vwm tests

git add .
git commit -m "Add my-new-thing"
git push -u origin feature/my-new-thing
```

Open a PR on GitHub from `feature/my-new-thing` -> `develop`. Review the
diff yourself (this catches mistakes), wait for CI green, merge.

### Fixing a bug

Same flow but use `fix/short-name` for the branch.

### Releasing a new version

```bash
git checkout develop
# Update CHANGELOG.md: move "Unreleased" entries into a dated version section.
# Update cocolab_vwm/_version.py: bump the version number.
git add CHANGELOG.md cocolab_vwm/_version.py
git commit -m "Bump to v0.x.y"

git checkout main
git merge --no-ff develop -m "Release v0.x.y"
git tag -a v0.x.y -m "Release v0.x.y"
git push origin main
git push --tags
```

Then on GitHub create a Release from the new tag with the changelog
section as the description.

## Adding a co-author or collaborator

In the repo settings -> Collaborators, invite by GitHub username. They'll
need to fork-and-PR if you keep `main` protected, or push directly to
`develop` if you give them write access.

## Hooking it up to your existing lab Git server (if any)

If Krea has its own GitLab / Gitea, you can add it as a second remote:

```bash
git remote add krea git@git.krea.edu.in:cocolab/cocolab_vwm.git
git push krea main develop --tags
```

Push to both with one command:

```bash
git push --all origin
git push --all krea
```

## Useful commands you'll reach for

```bash
# See what changed since the last tagged release
git diff v0.1.0..HEAD --stat

# See who touched a file last
git blame cocolab_vwm/core/dynamics.py

# Roll back if a commit broke something (locally only)
git reset --soft HEAD~1            # undo commit, keep changes staged
git reset --hard HEAD~1            # undo commit and discard changes

# Stash work-in-progress so you can switch branches
git stash
git checkout main
git stash pop                      # bring it back

# See your branches
git branch -a                      # local + remote
```

## Anti-patterns to avoid

1. **Don't commit large binary files** (.npz, .h5, .pkl). They bloat the
   repo permanently. The `.gitignore` already excludes these. Use a
   `results/published/` folder for figures only and put bigger artefacts
   on Zenodo or Krea's data store.

2. **Don't push to `main` directly.** Always go through `develop` and PRs.

3. **Don't squash-merge feature branches without saving the history.**
   The CHANGELOG is your release-level summary; the per-commit history
   is your debugging tool when something regresses.

4. **Don't change a public function signature without bumping the minor
   version**, even if no test breaks. Other people's scripts might.
