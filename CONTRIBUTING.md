<!--
Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
HYPRE Project Developers. See the top-level COPYRIGHT file for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)
-->

# Contributing to hypre

For any substantial contribution to the library, especially patches larger than 100 lines, please reach out to the hypre team for discussion before starting your own work. Early discussion helps us coordinate design choices, avoid duplicated effort, and point you to related ongoing work.

hypre is an open source project distributed under the terms of both the MIT license and the Apache License (Version 2.0). All new contributions must be made under both licenses. We welcome contributions via pull requests as well as questions, feature requests, or bug reports via issues. See [SUPPORT.md](./SUPPORT.md) or contact the team on the [hypre issue tracker](https://github.com/hypre-space/hypre/issues).

If you are not a member of the `hypre-space` organization, you will not have permission to push branches directly to the repository. In that case, start by creating a fork so you can publish your branch and open a pull request.

1. Create your branch from `hypre:master`.
2. Use clear branch names, commit messages, and pull request titles.
3. Write commit messages in the imperative mood, for example `Add ...`, `Fix ...`, or `Update ...`.
4. Keep commits logically organized, and keep pull requests focused on one topic when possible.
5. Describe the motivation for the change, any user-visible impact, and what testing you ran in the pull request description.
6. Review existing issues before opening a new one. Your topic may already be under discussion or development.
7. When reporting bugs or requesting enhancements, be explicit about the expected behavior, the observed behavior, how to reproduce the problem, and any relevant environment or build configuration details.
8. Add or update tests and documentation when they materially improve confidence in the change, especially for user-facing behavior, APIs, or build options.
9. Small fixes such as typos, minor documentation updates, or narrowly scoped cleanups can usually be opened directly without prior discussion.
