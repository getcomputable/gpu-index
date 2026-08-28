# CGI Governance

## Code license permanence

The pipeline code in this repository is and will remain under the Apache
License 2.0. Computable's commercial rights attach to the published index
values (LICENSE-DATA.md) and to the trademarks (TRADEMARKS.md), never to this
code. Contributions are accepted under the DCO because no relicensing option
is being reserved.

## Methodology changes

The methodology is versioned. Every published record embeds the full parameter
set that produced it. Any change to a parameter that touches a published day
requires minting a new methodology version; the calculation refuses to extend
a series under altered parameters. Prior versions stay frozen and readable.
Published values are never revised; corrections publish forward. Methodology
documents are published (METHODOLOGY.md and the per-lane configs under
config/), and the open pipeline runs those baked config files directly.

## Panel membership

Panel membership is the parameter; liveness weights within the panel are
computed (see METHODOLOGY.md section 8). Adding or removing a panel provider is a governance
decision taken at panel review, with the reasoning documented in the
methodology change log (CHANGELOG.md). Merging a provider's collector into
this repository does not by itself make that provider a panel member.

## Roles and contact

Maintainers review code and operate the published pipeline. Methodology and
panel changes are decided by Computable and recorded in the change log
(CHANGELOG.md) with effective dates. Questions and error reports: team@getcomputable.com. A
report about a published value is answered with the replay evidence for that
value.
