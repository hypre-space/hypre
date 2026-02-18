This document gives information on Continuous Integration (CI) in hypre.

# Continuous Integration Workflow

The hypre repository includes an intentionally simple GitHub Actions
workflow (`.github/workflows/ci.yml`) that provides a deterministic
Debug build and the `check` regression target on both Linux and macOS.
Understanding how this CI policy operates is essential when opening pull
requests (PRs), because the CI status is required before changes may be
merged into `master`.

## Trigger conditions

The workflow runs automatically for:

-   pushes to `master` (to protect the branch after merges),
-   all PR events (`opened`, `reopened`, `synchronize`) **and** label
    changes (`labeled` / `unlabeled`),
-   the manual `workflow_dispatch` action (a maintainer can explicitly
    re-run CI from the "Run workflow" button in the GitHub UI).

## Label requirement

Every PR must carry the label `Run CI` before the build begins. A
lightweight `require-label` job validates the label, and if it is
missing the workflow fails immediately with a clear message:

``` text
Add the 'Run CI' label to this pull request to trigger CI.
```

Because label changes are part of the trigger set, adding (or re-adding)
`Run CI` automatically restarts the workflow. Developers should
therefore:

1.  Draft the PR.
2.  Apply the `Run CI` label once the branch is ready for testing.
3.  Re-apply the label after pushing new commits if the workflow needs
    to be restarted (or use the manual dispatch button).

## Workflow structure

The following jobs run in sequence:

`require-label`

:   *Runs only on PR events.* This job inspects the label list and fails
    fast if `Run CI` is absent. The downstream build job is skipped when
    this happens, which keeps GitHub's check list concise.

`build-and-check`

:   A matrix job covering `ubuntu-latest`, `macos-latest`, and
    `windows-latest` (MSVC). Windows runs two variants: one with
    MS-MPI enabled and one with MPI disabled. Each worker:

    1.  Checks out the hypre sources.
    2.  Installs MPI runtime dependencies (`open-mpi` via `apt` on
        Ubuntu, `open-mpi` via Homebrew on macOS, and MS-MPI runtime +
        SDK downloaded from Microsoft's GitHub releases on Windows) so
        that `mpiexec` is available for MPI-enabled jobs.
    3.  Configures a Debug CMake build from `src` into `build` with
        tests enabled, and disables MPI for the Windows sequential
        variant.
    4.  Builds the library with `cmake --build build --parallel`.
    5.  Invokes `cmake --build build --target check` to run simple tests
        using each of the three conceptual interfaces in hypre: IJ,
        Struct, and SStruct.
    6.  Optionally, you can download GitHub Actions runner logs to help
        diagnose failures.

## Helpful GitHub links

When you need background on GitHub Actions terminology or you want to
inspect a specific hypre workflow run, the following references collect
the most useful starting points:

-   **Understanding GitHub Actions** -- GitHub\'s introductory guide
    explains the terminology (workflows, jobs, runners, etc.) at
    [Understanding GitHub
    Actions](https://docs.github.com/en/actions/get-started/understand-github-actions).
-   **Understanding GitHub Continuous Integration** -- GitHub\'s CI
    overview describes how event triggers and test automation work in
    practice: [Understanding CI with GitHub
    Actions](https://docs.github.com/en/actions/get-started/continuous-integration).
-   **Writing workflows** -- GitHub\'s guide on using workflow templates
    explains how to create and customize workflows: [Using workflow
    templates](https://docs.github.com/en/actions/how-tos/write-workflows/use-workflow-templates).
-   **Actions dashboard** -- [Actions
    dashboard](https://github.com/hypre-space/hypre/actions) shows the
    full history of workflow runs. Filter by branch or event and click a
    row to open detailed logs.
-   **Specific workflow page** -- [CI workflow
    page](https://github.com/hypre-space/hypre/actions/workflows/ci.yml)
    focuses on the main CI workflow currently in place. Use the \"Run
    workflow\" button for a manual `workflow_dispatch` when needed.
-   **Pull request checks tab** -- Each PR includes a \"Checks\" tab
    that aggregates the `require-label` and matrix results. Expand a
    failing job to view the captured console output or re-run a subset
    of jobs. Example: [PR #1426 checks
    page](https://github.com/hypre-space/hypre/pull/1426/checks).
