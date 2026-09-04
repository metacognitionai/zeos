# Tabulator, vendored

Version **6.5.2**, MIT (see `LICENSE`). Fetched from unpkg:

    curl -sSfL -o tabulator.min.js  https://unpkg.com/tabulator-tables@6.5.2/dist/js/tabulator.min.js
    curl -sSfL -o tabulator.min.css https://unpkg.com/tabulator-tables@6.5.2/dist/css/tabulator.min.css
    curl -sSfL -o LICENSE           https://unpkg.com/tabulator-tables@6.5.2/LICENSE

sha256 as fetched, before the edit below:

    04802e757fa4189342c666d0f970a01d761c312798f31ffc664c24cbccc7ce3e  tabulator.min.js
    b55e204b2f968cecc4d3663d37858093b31dd22d20f01d76f590726ee18f7e1f  tabulator.min.css

**Two edits to each file.** The trailing `sourceMappingURL` comment is
stripped: the maps are not shipped, and `web/server.py` inlines these files into
the page, so the reference would resolve against the page's own URL and 404. The
page is required to work with no network at all -- `viewer.css` says so at the
top -- and a dangling reference is the one thing that would make that untrue.
And a trailing newline is added, because the repository's `end-of-file-fixer`
hook runs over vendored files too.

## Why it is here rather than a dependency

Committed, not fetched at build time or linked from a CDN, for the reason the
whole page is one inlined file: `viewer --export` has to open from a filesystem
with the network off. `packages = ["src/zeos_space_invaders"]` ships everything under
the package, so these travel with the wheel the way `players/prompts/` does.

Only the index and the comparison tables use it. The run page does not, and
`page(vendor=False)` leaves it out of an exported run -- see `web/server.py`.

## Upgrading

Re-run the three `curl` lines with a new version, strip the two
`sourceMappingURL` comments, **add the trailing newline `LICENSE` ships
without** -- the repository's `end-of-file-fixer` hook runs over vendored files
too, and a hook that rewrites a file on checkout is worse than the one byte --
update the version and hashes above, and check
`grep -c 'url(' tabulator.min.css` is still `0`: a stylesheet that reaches for an
image would break the offline invariant. Then bump the `persistenceID` suffixes
in `viewer.js` if the column set moved with it.
