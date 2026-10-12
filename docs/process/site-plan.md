# Serenity project site

Status: site and R2 media complete, 2026-10-12. Pages domain verified with
HTTPS enforced; deployment workflow ready for the first merge to main.

## Purpose and public state

Give interested engineers a visual introduction to Serenity and an accessible
path into its GPU work and milestone writeups. Explain the motivation: rendering
offers another way to investigate streaming computation, small running state,
and reuse, techniques that also matter to inference.

The official public state is milestone 1. ReSTIR is in progress on a branch.
Internal implementation notes do not imply a publicly completed milestone.
The 60 fps goal remains a goal. Historical measurements retain their source,
machine, configuration, and date; video playback rate is not render throughput.

## Site and hosting

- GitHub Pages hosts the static site at `serenity.plainsight-systems.com`.
  Hugo adapts Cradle's article presentation and build validation.
- Cloudflare R2 hosts selected renders, video, and originals in
  `plainsight-public-media`, served at `media.plainsight-systems.com`.
  All four public URLs are verified and configured in the media manifest.
- Render at the quality and resolution the work requires. Preserve original
  stills and masters. Offer a high-quality full-resolution video, a smaller
  playback option, and original downloads where useful.
- Encoding is judged against the original, especially moving fireflies, dark
  gradients, noise, and reflections. A file-size target must not silently
  remove the visual evidence the project is showing.
- R2 storage and delivery are separate from browser decoding and connection
  speed. Store URLs, dimensions, captions, and provenance in one media manifest.
  Use a poster and user-initiated playback, without downloading a master on load.
- R2 supports objects of roughly 5 TiB and free internet egress; storage and
  operations beyond its allowance are billable. Standard CDN plans cache up to
  512 MB per file. Larger originals can remain available for download; segmented
  delivery is an option if long high-bitrate playback needs it.

Checked 2026-10-12: [R2 limits](https://developers.cloudflare.com/r2/platform/limits/),
[pricing](https://developers.cloudflare.com/r2/pricing/),
[R2 caching](https://developers.cloudflare.com/cache/interaction-cloudflare-products/r2/).

## First release

### Shared visual family

Refined by the user 2026-10-12: keep general styling consistent with current
Cradle and Plainsight Systems. Full-width navy headers and footers (#0e1524),
ivory reading surfaces (#fafaf5), warm light text (#f1ebdc), brass (#d4ad5a),
self-hosted Plex Serif and Plex Mono, and the same uppercase navigation style.
Include a clear "A Plainsight Systems project" footer link. Keep Serenity's
illustration, render presentation, and editorial composition specific to it.

Plainsight Systems is the hub; the individual project sites are spokes. Omit
"Other work" and sibling-project lists from Serenity. Its About page links to
the Plainsight hub for the wider project and research context, as Cradle does.
Use Cradle's footer layout and content directly, substituting Serenity for the
project name: project and parent link above, Andrew P Hunter writing copyright
and linked CC BY 4.0 / Apache 2.0 licenses below. Keep its spacing and type sizes.
Apply the installed `andrew-voice` skill to public copy: explain the motivation
and mechanism in connected prose, with fewer slogan-like headings. The approved
Cradle passage is the calibration; current milestone records supply the facts.

The current Cradle template and live Plainsight computed styles were compared.
For production, keep these shared choices in a small explicit token set and
matching header/footer partials; a cross-repository theme package is not needed
for this first site. Changes in either reference should be reconciled deliberately.

Homepage, About page, writeups index, and milestone-one reading page. The homepage contains:

1. Existing screenprint artwork, title, motivation, author, and main links.
2. Actual milestone-one footage with a clear offline-render caption.
3. "Where we started": the original October 8 brass sphere and two fireflies.
4. The GPU questions that connect the renderer to inference.
5. The milestone-one writeup and a concise account of what is next.
6. One general GitHub link in the footer, alongside attribution and licenses.

Navigation refinement: GitHub belongs in the footer. Omit the header GitHub
link and the redundant homepage "View on GitHub" / "Build it locally" section.
Contextual links to a particular writeup remain beside the relevant prose.

Keep `docs/writeups/` authoritative. Publish explicitly selected articles and
assets. Link to research and architecture in GitHub initially. Do not publish
the private brief or turn the internal queue into the public status page.

## Mockup for approval

Width refinement (2026-10-12): use more of the screen throughout. One fluid
container aligns the header, footer, homepage sections, media, and About page.
Use Cradle's `clamp(1.25rem, 4vw, 4rem)` side gutters, without Serenity's
previous 1248/1360-pixel outer caps or 720-pixel About inset. About prose fills
the container, as Cradle's standalone pages do, with 20-pixel desktop body type.
Unlike Cradle's current 76rem outer cap, Serenity's container stays fluid to
honor the request for more screen use; narrow screens retain 20-pixel gutters.

`site-mockup/index.html`, `site-mockup/about.html`, and `site-mockup/style.css`
are a local, responsive prototype, not the production Hugo implementation. Real links go to
existing repository content. The film plays a local existing encode; R2 is not
configured. No simulated download or quality-selection controls are included.

The mockup uses local ignored media and must be served from the repository root:

```sh
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `/docs/process/site-mockup/`. The October 8 preview image and October 10
1080p v3 movie under `media/` are needed for the complete local preview. Fonts
are copied from Cradle, retain their SIL license, and are self-hosted.

Approval covers composition, visual direction, copy, and content order. After
approval: implement Hugo pages, prepare and verify publication media, configure
R2, integrate final URLs, check responsive and keyboard behavior, validate links
and publication boundaries, and configure the Pages deployment.

Mockup verification (2026-10-12): visually checked in the local browser at
390, 908, and 1440 pixels wide, with no horizontal overflow at the checked
widths. Confirmed local video playback, loaded images, render navigation, and
all local asset and anchor targets. Desktop and mobile review screenshots are
in `.cache/serenity-mockup-desktop.jpg` and `.cache/serenity-mockup-mobile.jpg`.
The local video is H.264, 1920 x 1080, 30 seconds, 30 fps playback, verified
with ffprobe. Publication encodes and R2 delivery remain future work.


## Implementation, 2026-10-12

The user approved building the site and rewriting the milestone-one article
with the voice guide, then addressing infrastructure. `web/` implements Home,
Writing (a milestones index that opens individual articles), Milestone 1, and
About, plus a 404. `docs/writeups/milestone-1.md` remains the article source.
It now develops the motivation and experiments in connected prose and states
what the references and historical measurements actually establish.

`scripts/site.py` stages only selected content, pins Hugo 0.154.2, validates
links/fragments/assets and page boundaries, and serves a loopback preview.
Self-hosted fonts, responsive tables, contents links, canonical/social metadata,
and project-subpath navigation are implemented. There is no client JavaScript.
Seven publication tests pass, including deliberately broken links, missing
assets, an unselected page, missing hosted media, and project-subpath routing.

The local preview runs at `http://127.0.0.1:1314/`. Both the project-subpath build
and root preview have passed the generated-site checks. Browser verification
covers desktop/mobile navigation, the article contents, layouts, image loading,
and actual 1920 × 1080 video playback. Screenshots are under `.cache/`.

Remaining infrastructure work, in order:

1. Configure the R2 bucket and media domain; select and upload the film, original
   brass image, optional smaller encode, and original-render download.
2. Verify uploaded bytes, content types, byte ranges/seeking, caching, and image
   and video appearance. Preserve originals; choose playback quality by looking
   at the dark gradients and moving fireflies, not an arbitrary size cap.
3. Set final HTTPS URLs in `web/data/media.json`. The public build deliberately
   refuses missing film/brass URLs. Preview-only media never enters that output.
4. Configure Pages and a workflow with pinned Hugo/Python and source verification;
   upload only `.cache/site/public/`. Confirm the eventual base URL and domain.
5. Verify the deployed pages, media, social previews, 404, and mobile playback.

No bucket, DNS, Pages setting, or deployment has been changed in this phase.


## R2 upload progress, 2026-10-12

The user created `plainsight-public-media` and connected a custom domain in
Cloudflare. Its exact hostname has not yet been supplied. The 1080p film,
540p alternative, and original brass PNG were already uploaded at the bucket
root, without the proposed `serenity/milestone-1/` prefix. Keep those existing
keys when wiring the site.

The master, `2026-10-10-marbles-opening-v3-master.mp4`, was then uploaded beside
them using the existing AWS CLI and bucket-scoped local profile `serenity-r2`.
The upload used multipart transfer, `Content-Type: video/mp4`, and
`Cache-Control: public, max-age=31536000, immutable` for this versioned filename.
A complete streamed readback verified all 416,036,234 bytes against the local
SHA-256: `4f7751d49d51f18e9c235267c9a6fef349fcc9b7bc51f165d869f0427f99993e`.
Credentials are stored outside the repository. Access objects directly by
bucket name; account-wide bucket listing is denied by this scoped token.

Remaining: confirm public hostname and delivery, integrate media URLs, and set
up Pages deployment. No Pages deployment has occurred.


## Public media integration, 2026-10-12

Confirmed the suggested custom hostname, `media.plainsight-systems.com`.
All four root-level object URLs return the expected Content-Length and MIME
type. Tail-range requests return 206 and match the corresponding local bytes;
the 1080p film also showed a CDN cache HIT. The initial Python urllib probes
were rejected with Cloudflare error 1010 (client signature); standard curl
requests succeeded. No security setting was changed.

The manifest now uses these URLs. Smaller-playback and original-master links
are visible, and builds do not require local media. The missing-media test now
explicitly clears URLs in its fixture so it continues testing that error path.
Public build and preview checks pass. GitHub Pages configuration and deployment
remain outstanding.

Browser verification also passed: the main video played from the public custom
hostname at 1920 × 1080, the brass PNG loaded remotely, and clicking the video
timeline sought to approximately 15 seconds with readyState 4. Both alternate
file links are visible. Evidence: `.cache/serenity-r2-media.jpg`.


## Pages deployment preparation, 2026-10-12

The user configured `serenity.plainsight-systems.com` in Pages and DNS. GitHub
reports the domain verified, its certificate approved, and HTTPS enforced.
The Hugo and build-script defaults now use that HTTPS address.

The workflow runs the seven publication checks and builds on PRs to main.
Only a push or manual run on main can upload the public output and deploy it.
Actions are pinned by commit, Python to 3.14.5, and Hugo to 0.154.2 with the
same checked Linux archive as Cradle. No renderer build or R2 credentials are
needed. Local tests and the production build pass.

The site is being packaged on a separate branch based on main. Only the site,
approved writeup, and related records are included; ReSTIR stays on its branch.
First publication follows merge, then live page and media verification.
