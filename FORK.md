# Fork maintenance

This fork preserves a small SQLite vec0 query compatibility patch on the upstream v0.23.2 base.

- `main` mirrors `basicmachines-co/basic-memory:main`.
- `fix/sqlite-vec-source-hash` contains the patch and its focused regression test.
- Consumers install an immutable commit from the patch branch, never a moving branch name.

Synchronize the upstream mirror with:

```sh
gh repo sync bassfredes/basic-memory --source basicmachines-co/basic-memory --branch main
git fetch origin
git fetch upstream --tags
```

To adopt a newer upstream release, create a new maintenance branch at that release and cherry-pick the patch commit if the fix is still needed. Run the regression and relevant upstream checks before updating consumer lock files. Do not force-push the branch currently referenced by deployed installations.

The regression calls SQLiteVecIndex.search against a real sqlite-vec database and checks that current vectors are returned while stale generations and other projects are excluded.

No user notes, client configurations, credentials or machine-specific integration files belong in this public fork.
