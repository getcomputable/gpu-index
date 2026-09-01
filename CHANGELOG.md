# Methodology change log

The methodology change log referenced by [GOVERNANCE.md](GOVERNANCE.md).
Entries record lane mints, retirements, and adopted methodology changes,
with effective dates. Every published record embeds the full parameter set
that produced it; any parameter change touching a published day mints a
new methodology_id, and prior versions stay frozen and readable under
their own keyspaces. Newest first.

## 2026-09-01

- Change log restarted at the public launch of this repository. Earlier
  internal iterations are superseded by the published record itself:
  every published observation names its `methodology_id` and embeds the
  complete calculation parameters that produced it, so any historical
  print remains verifiable with `./reproduce` regardless of when its
  methodology version was minted. Changes from this date onward are
  recorded here.
