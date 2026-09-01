# Computable GPU Index (CGI) Data License

This file governs the published CGI index values and data artifacts (JSON, CSV,
and similar files published through data.getcomputable.com,
api.getcomputable.com, mcp.getcomputable.com, and getcomputable.com/gpu-index,
including historical archives and latest pointers), together the
"Index Values". It does not govern (a) source code in this repository, which is
under the Apache License 2.0 (see LICENSE), (b) collected raw observations
outside the published panel record, for which all rights are reserved, or
(c) the CGI name and logo (see TRADEMARKS.md).

## 1. Base license: CC BY-NC 4.0

The Index Values are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License.
https://creativecommons.org/licenses/by-nc/4.0/

Attribution format: "Computable GPU Index (CGI), (c) 2026 Computable,
https://github.com/getcomputable/gpu-index, licensed CC BY-NC 4.0". Cite the
artifact date and sha256 where practical.

Receipts embedded in the artifacts report third-party provider prices as
facts, included solely for verification and provenance. Computable claims no
rights in the underlying provider prices; the license above covers
Computable's rights in the selection, arrangement, computed index values, and
the database they form.

## 2. Settlement and Product Use: separate license required

The license above does NOT permit, and Computable reserves all rights in, the
following uses. Each requires a separate written license:

  a. using any Index Value as the basis of, or a component of, a financial
     product, including determining the settlement value, strike, trigger,
     margin, or payout of any futures contract, option, swap, forward,
     structured product, or fund;
  b. calculating or disseminating NAV or iNAV of a fund or similar vehicle;
  c. non-display use in pricing, quoting, risk, or execution systems operated
     as a commercial service;
  d. redistribution of the Index Values, at any delay, as part of a paid data
     product or feed;
  e. naming a product or index with the CGI mark (see TRADEMARKS.md).

For a Product Use license contact team@getcomputable.com.

## 3. Integrity and provenance

Artifacts are immutable once published and are committed to by sha256.
Republication must not alter values; corrections are issued as new dated
artifacts under a new methodology version.

## 4. No warranty; no advice

The Index Values are provided "as is", without warranty of any kind. Computable
does not guarantee accuracy, completeness, or continuity of publication and has
no liability for errors, omissions, or interruptions. Nothing here is
investment advice. Products referencing the Index Values are not sponsored,
endorsed, sold, or promoted by Computable unless separately agreed in writing.

## 5. Machine-readable declaration

Each artifact embeds a `license` block with these three fields:

    "license": {
      "spdx": "CC-BY-NC-4.0",
      "url": "https://creativecommons.org/licenses/by-nc/4.0/",
      "attribution": "<index name> by Computable — <index domain>"
    }

`spdx` is pinned to `CC-BY-NC-4.0`; `url` is the CC BY-NC 4.0 license
text; `attribution` names the index and Computable. A fourth field,
`commercial_licensing`, is reserved for the Section 2 contact and is not
published today; readers accept it when present, so adding it later
breaks nobody, and until then Section 2 carries that address. The
operative terms are this file, not the embedded field.
