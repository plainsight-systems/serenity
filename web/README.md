# Serenity project site

Hugo implements the homepage, Writing index, both milestone articles,
and About page. The shared navy framing and Plex typography follow Cradle and
Plainsight Systems. No JavaScript, analytics, external fonts, or renderer build
is required. A custom 404 page is included.

## Run locally

Requires Python 3.10+ and **Hugo 0.154.2** (the build checks the version).
From the repository root:

```sh
python3 scripts/site.py serve --port 1314
```

Open <http://127.0.0.1:1314/>. Rerun after source edits; this is a build followed
by a static server, not a watcher. The server binds only to loopback and serves
only the generated output. Port 1313 is the default if `--port` is omitted.

The ReSTIR film, side-by-side comparison, milestone-one film, their smaller
playback options and original masters, and the brass sphere are hosted at
`media.plainsight-systems.com` in R2 bucket
`plainsight-public-media`. Their URLs are configured in `data/media.json`, so
normal builds and previews do not require the ignored local media files.

For offline media development, clear the relevant manifest URLs and pass
`--local-media`. That mode copies only the explicitly selected local files to
`.cache/site/preview/`. It never copies all of `media/`; do not deploy that output.

## Content and publication boundary

- `docs/writeups/milestone-1.md` and `milestone-2.md` are the authoritative
  articles. The builder stages each as a Hugo leaf bundle, removing its
  duplicate H1 and supplying publication metadata from `ARTICLES` in
  `scripts/site.py`. Edit the prose in the original writeup; update its registry
  entry if its title changes. The registry also selects its images and film.
- The homepage features the ReSTIR film. Milestone 2 includes the comparison
  film; milestone 1 retains its original film. Each article has its own Writing
  thumbnail and social preview. Relative links between writeups resolve to
  published pages; relative research links resolve to GitHub.
- `content/` holds the About page and Writing introduction.
- `layouts/` holds the shared shell, homepage, and reading layouts.
- `assets/site.css` holds the common tokens and responsive styles.
- `static/fonts/` contains the self-hosted Plex fonts and their SIL license.
- The project illustration and article stills are explicitly copied from
  `docs/assets/` and `docs/writeups/images/`. Each article still is also available
  at a stable path for link previews and the Writing index.

No directory mount exposes `docs/`, `media/`, research, process records, or the
private brief. Research links go to named files in GitHub. Adding a milestone
requires explicitly adding its source and metadata to the staging allowlist,
then updating the expected public pages and verification. Future milestones
are not generated from the internal queue.

## Media configuration

`data/media.json` records the source file, hosted URL, caption/provenance, and
known dimensions for selected media. Leave `local` as the source record and
set `url` to the final HTTPS object URL after upload and delivery checks.

| Entry | Public build behavior |
|---|---|
| `artwork`, `poster`, `baseline_poster`, `comparison_poster` | Site-owned assets by default; may use hosted URLs |
| `film`, `baseline`, `comparison`, `brass` | Hosted URLs required; local files only with `--local-media` |
| Each film’s `_small` and `_original` entries | Optional explicit links, shown only when URLs are set |

Each film has native controls, a poster, and `preload="none"`; it does not
start automatically or download the original render on page load. The smaller
file opens separately so the choice remains usable without JavaScript.
No encoding or resizing occurs during the site build.

Milestone-two objects use their dated filenames at the bucket root. Both the
ReSTIR film and the comparison have playback, smaller, and master versions.
The comparison’s `1080p30-16mbps` filename contains a **1920 × 540** video;
its smaller version is 960 × 270 and its master is 3840 × 1080. The player
uses the actual dimensions. ReSTIR playback and master are 1920 × 1080,
with a 960 × 540 smaller version. All six are 30 seconds at 30 fps playback.

New uploads use `Content-Type: video/mp4` and
`Cache-Control: public, max-age=31536000, immutable`. Use new filenames for
changed media. The local `serenity-r2` AWS CLI profile is scoped to
`plainsight-public-media`; credentials stay outside this repository.

The 892 MB comparison master exceeds the standard CDN cache limit and is
served from R2 without edge caching. The page plays the smaller file by
default and provides the master separately. See
[Cloudflare’s R2 caching limits](https://developers.cloudflare.com/cache/interaction-cloudflare-products/r2/).

## Build and verify

```sh
# Build with hosted media for https://serenity.plainsight-systems.com/.
python3 scripts/site.py build

# Publication-boundary and broken-link tests, with real Hugo output.
# These do not need ignored media or network access.
python3 -m unittest discover -s tests -p test_site.py -v

# To override the intended public address:
python3 scripts/site.py build --base-url https://plainsight-systems.github.io/serenity/
```

A public build writes `.cache/site/public/`. Missing required hosted URLs are
an explicit error. Every build checks local links, fragments, assets, duplicate
IDs, basic page landmarks, and the exact set of public pages. The tests also
break fragments, paths, assets, and publication boundaries to prove the checks
reject them. A failed build leaves the preceding output available.

The HTTPS media URLs are checked for syntax, not fetched by the build. Delivery,
content types, byte-range seeking, caching, and full-quality playback must be
verified against the actual host during the infrastructure phase.

## GitHub Pages

The public address is <https://serenity.plainsight-systems.com/>. GitHub Pages
uses GitHub Actions, with the custom domain verified and HTTPS enforced.

`.github/workflows/pages.yml` tests and builds pull requests. A push to `main`
(or a manual run on `main`) also publishes only `.cache/site/public/` through
the `github-pages` environment. Other branches cannot deploy. Actions are
pinned by commit; Python is pinned to 3.14.5, and the Hugo 0.154.2 download is
checked against its SHA-256. R2 credentials are not needed by CI.

The deployment runs independently of the native renderer build. Adding a new
milestone remains an explicit content change; merging renderer work does not
make its research notes public site pages.
