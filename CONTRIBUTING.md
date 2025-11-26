# Contribution Guide 🚀

Welcome to the **SMLE** project! This guide explains how to prepare your changes, work with Git, and open good pull requests.

## Initial Checks

Before you start coding, read the main `README.md` and the [documentation](https://pysmle.org/) to understand the project goals, basic usage, and current roadmap. If you are unsure what to work on or want feedback on an idea, start a conversation on the project’s Discord server or in the GitHub discussion thread linked from the README so the maintainers can help you scope a useful contribution.

Whenever possible, prefer small, focused changes over very large pull requests.
If you plan a bigger feature or refactor, discuss it first to confirm that it fits the project direction.

## Basic Git Workflow

If you are new to Git and GitHub, the steps below describe a simple way to contribute:

- Fork the repository to your own GitHub account.

- Clone your fork locally:
  `git clone https://github.com/<your-username>/smle.git`
- Create a new branch for your work:
  `git checkout -b feature`
- Make and test your changes.
- Stage your changes:
  `git add <files>`
- Commit with a clear message:
  `git commit -m "Describe your change"`
- Push your branch:
  `git push origin feature`
- Open a pull request from your branch into the main `smle` repository.

Try to keep each branch focused on a single issue or feature so that reviews are easier.

## Making Changes and Opening a PR

When you make changes, aim to:

- Follow the existing code style and structure where possible.
- Add or update tests and documentation if your change affects behavior or public APIs.
- Keep commits logically grouped (for example, separate “refactor” from “new feature” where it makes sense).

Once your branch is ready:

- Ensure the project still installs and runs as expected (and tests pass, if available).
- Rebase or merge the latest default branch into your branch to resolve conflicts before opening the PR.
- Push your final changes and prepare to open a pull request.

## Submitting a Pull Request (PR)

1. From your fork on GitHub, use the “Compare & pull request” button to create a PR.
2. Use a clear, descriptive title and fill out the PR description so reviewers understand what you changed and why.
3. If you are working on a labeled issue (for example, “good first issue”), reference the issue number in the PR description (for example, “Fixes #123”).
4. Submit the PR and respond to review comments; you may be asked to adjust code, tests, or documentation before merge.

## Reporting Issues and Getting Help

If you find a bug or have a feature request but do not plan to implement it yourself:

- Open a GitHub issue with a clear description of the problem or idea.
- For bugs, describe the expected behavior, the actual behavior, and how to reproduce it (include versions and environment details when possible).
- For feature requests, explain the use case and how it would benefit SMLE users.

If you need guidance at any point, use the Discord server or the linked GitHub discussion thread to ask questions and coordinate with maintainers and other contributors.