# WP5 Retrieval Preparation Notes

WP5 adds the first governed retrieval-preparation layer for the Salifort PACE Navigator.

What WP5 produces:

- a deterministic retrieval-pack builder
- a retrieval-pack validator
- a governed eligibility policy
- normalized retrieval documents and structure-aware chunks
- stable provenance back to source paths and registry entities

What this pack is for:

- future API-backed retrieval
- future embedding generation and vector indexing
- future governed answer assembly
- future citation-aware navigator behavior

What this pack is not:

- not a retrieval runtime
- not an embedding index
- not a vector database
- not an API integration
- not a chatbot

Why this is still non-invasive:

- it does not change app behavior
- it does not change modeling or artifact-building logic
- it does not change preserved public truth
- it builds only deterministic local preparation artifacts

What later phases can safely build on:

- embedding generation
- vector indexing
- governed chunk retrieval
- answer assembly with provenance and drift awareness
