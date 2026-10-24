# Attribution and publication scope

The manuscript is credited, at the repository owner's direction, to
**GPT-6.1sol Max & Hermes**. The names identify agent-generated research output;
the host account is the publisher, not an additional manuscript author.
This is a public research manuscript, not a claim of journal publication or
peer review.

Original project code is MIT-licensed. The manuscript, original figures and
numerical records in `paper/`, `docs/` and `data/` are shared under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), matching the
publisher's blog convention. Cite the manuscript attribution and link to the
repository when reusing these materials. Third-party works retain their own
licenses. The bundled KaTeX fonts and CSS have their MIT notice in
`paper/assets/katex/LICENSE`.

No GLoW/GSL binaries, third-party source trees, literature full-text caches,
credentials, private keys, environments or internal editorial records are
redistributed. GLoW is an external comparison, not our implementation:
<https://github.com/glow-astro/GLoW>.

`SOURCE_MANIFEST.json` records original project-relative names and byte hashes.
Compressed numerical files decode to the original bytes. One saved native
backend record replaces a private home-directory prefix with `${HOME}`;
a saved interpreter location is represented as `${VIRTUAL_ENV}/python`.
All numerical and boolean values are unchanged, and both transformations are
explicit in the manifest. No benchmark was rerun to prepare this release.

The portable test selection retains the original numerical unit tests.
Historical study/analysis command-line tests and machine-specific native
GLoW tests are outside this selection. The new public replay checks saved
whole-band results and recorded paired timings without running the native
competitor or creating a new reference grid.
