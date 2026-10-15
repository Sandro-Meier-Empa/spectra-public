# Contributing

## Pre-commit hooks

This repository uses [pre-commit](https://pre-commit.com/) to run code quality
checks before commits. Before you can make any commits, set up the pre-commit
hooks:

```bash
pip install pre-commit
pre-commit install
```

One of the hooks is [`pyright`](https://github.com/microsoft/pyright). This is a
static type checker for Python that can help catch type errors before they
occur. Since pyright also resolves imports, it must have access to the virtual
environment where the package and its dependencies are installed. To make sure
pyright can find the virtual environment, create a `pyrightconfig.json` file in
the root of the repository with the following content:

```json
{
  "include": [
    "src",
    "scripts"
  ],
  "typeCheckingMode": "standard",
  "venvPath": "/path/to/your/venv/parent/directory",
  "venv": "name_of_your_venv"
}
```

The pre-commit hooks run automatically before each commit. If a hook fails, the
commit is aborted. Fix the reported problem, stage the changed files again with
`git add`, and rerun `git commit`.

## Troubleshooting
Sometimes pre-commit reports `unstaged files detected`. This can happen when a
file has staged changes and additional unstaged changes in the same file. It can
also appear after a hook such as pyright raises an error. In that case, check the output of git log to see the error message or check the current state with `git status`, fix the issue, stage the intended final state with `git add`, and commit again.

Sometimes, pyright reports an error which you just fixed, but the error still appears in the pre-commit output. In that case, check if you have staged the fixed file with `git add`. If the error persists, it can be that pyright uses a cached version of the file. In that case, you can clear the pre-commit cache and run pyright again.
```bash
pre-commit clean
pre-commit run pyright --all-files
```

## Releasing a new version

Package versions follow [Semantic Versioning](https://semver.org/):

- `patch` for backward-compatible bug fixes
- `minor` for backward-compatible functionality
- `major` for incompatible API changes

Git tags named `vX.Y.Z` are the source of truth for package versions. Development
builds after a release receive a PEP 440 development version containing their
distance from the latest tag and Git commit identifier.

Before releasing, update `main` and fetch all release tags:

```bash
git switch main
git pull --ff-only --tags origin main
```

Preview the next version without modifying the repository:

```bash
bump-my-version bump patch --dry-run -vv
```

Replace `patch` with `minor` or `major` when appropriate. To run the configured
pre-commit and test checks and create the annotated release tag:

```bash
bump-my-version bump patch
```

Inspect the tag before publishing it:

```bash
git show v1.0.2
```

Then push that exact tag:

```bash
git push origin v1.0.2
```

Use the version reported by Bump My Version in place of `v1.0.2`. Pushing is
deliberately separate from tag creation so that the release can be reviewed
locally first. The release command refuses to run with uncommitted changes.
