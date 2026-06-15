## Summary

What does this PR do and why?

## Related issue

Closes #<issue-number>

## Changes

- 

## Testing

- [ ] Ran the test suite: `python -m unittest discover -s tests` (all passing)
- [ ] Added or updated tests covering this change

## Invariant checklist

Confirm this change does not silently break the core contracts (or explain the migration if it does):

- [ ] Fusion contract unchanged (`2048 + 32 + 28 = 2108`), or checkpoints/migration handled
- [ ] Phase 3 / Phase 4 still use `pairing_mode="adjacent_cache"`
- [ ] Fake-positive label convention (`fake = 1`) preserved
- [ ] No internal/private notes, secrets, or large binaries added

## Additional notes

Anything reviewers should pay special attention to (metric changes, perf, etc.).
