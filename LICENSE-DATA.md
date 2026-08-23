# Computable GPU Index (CGI) Data License

This file governs the published CGI index values and data artifacts (JSON, CSV,
and similar files published through the distribution endpoints listed in the
README, including historical archives and latest pointers), together the
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

## 2. Additional permission: Derived Index Grant

As an additional permission granted on top of CC BY-NC 4.0 (this section can
only broaden the CC license, never narrow it), Computable grants you a
worldwide, royalty-free, non-exclusive license to create Derived Works from the
Index Values and to publish them, including in commercial publications,
research, and commercial data products, provided that ALL of the following
hold:

  a. Publication. The Derived Work, or the portion computed from Index Values,
     is available to the public at no charge for at least viewing.
  b. Attribution. The Derived Work carries the Section 1 attribution statement
     and states what was modified or computed.
  c. No reconstruction. The Derived Work cannot be reverse engineered to
     recover the Index Values series and is not a substitute for the Index
     Values or for any unpublished Computable data.
  d. No product use. The Derived Work is not used for any purpose in Section 3
     without a separate license.

"Derived Works" means any data or information resulting from modification,
adaptation, compilation, evaluation, or any other recasting or processing of
the Index Values, alone or together with other data.

## 3. Settlement and Product Use: separate license required

The licenses above do NOT permit, and Computable reserves all rights in, the
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

For a Product Use license contact index@getcomputable.com.

## 4. Integrity and provenance

Artifacts are immutable once published and are committed to by sha256.
Republication must not alter values; corrections are issued as new dated
artifacts under a new methodology version.

## 5. No warranty; no advice

The Index Values are provided "as is", without warranty of any kind. Computable
does not guarantee accuracy, completeness, or continuity of publication and has
no liability for errors, omissions, or interruptions. Nothing here is
investment advice. Products referencing the Index Values are not sponsored,
endorsed, sold, or promoted by Computable unless separately agreed in writing.

## 6. Machine-readable declaration

Each artifact embeds:

    "license": {
      "values": "CC-BY-NC-4.0",
      "grant": "LICENSE-DATA.md in the CGI repository",
      "attribution": "Computable GPU Index (CGI), (c) 2026 Computable"
    }

The operative terms are this file, not the embedded field.
