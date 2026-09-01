// Envelope + digest + pretty-print EXACTLY as the publisher pipeline
// writes them: a byte-exact Node mirror of its envelope creation, file
// encoding, payload digest, and key-sort behaviors.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

function sortJson(value) {
  if (Array.isArray(value)) return value.map(sortJson);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, sortJson(entry)]),
  );
}

function digestPayload(payload) {
  return createHash('sha256')
    .update(JSON.stringify(sortJson(payload)))
    .digest('hex');
}

function createArtifactEnvelope(options) {
  const observations = options.data.observations;
  const stamps = observations.map((o) => o.observed_at).sort();
  const generated = observations.map((o) => o.generated_at).sort();
  const generatedAt = generated.at(-1);
  const disclosureRestatementCount = observations.reduce(
    (count, o) => count + ('restatements' in o ? o.restatements.length : 0),
    0,
  );
  const payload = {
    data: options.data,
    meta: {
      schema_version: 1,
      index_name: options.indexName,
      generated_at: generatedAt,
      from_observed_at: stamps[0] ?? null,
      to_observed_at: stamps.at(-1) ?? null,
      observation_count: observations.length,
      disclosure_restatement_count: disclosureRestatementCount,
    },
    license: {
      spdx: 'CC-BY-NC-4.0',
      url: 'https://creativecommons.org/licenses/by-nc/4.0/',
      attribution: `${options.indexName} by Computable — ${options.indexDomain}`,
      commercial_licensing: options.commercialLicensingUrl,
    },
  };
  return { artifact_sha256: digestPayload(payload), ...payload };
}

function encodeArtifact(envelope) {
  return `${JSON.stringify(sortJson(envelope), null, 2)}\n`;
}

const [specPath, outRoot] = process.argv.slice(2);
const spec = JSON.parse(readFileSync(specPath, 'utf-8'));
for (const file of spec.files) {
  const envelope = createArtifactEnvelope({
    data: file.data,
    indexName: spec.indexName,
    indexDomain: spec.indexDomain,
    commercialLicensingUrl: spec.commercialLicensingUrl,
  });
  const target = join(outRoot, file.path);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, encodeArtifact(envelope));
  console.log(`${envelope.artifact_sha256}  ${file.path}`);
}
