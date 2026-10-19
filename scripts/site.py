#!/usr/bin/env python3
"""Build Serenity's explicitly selected public pages with pinned Hugo.

Contract: docs/writeups is authoritative; only the named milestone articles and
selected media enter staging. Process documents, private planning, and the rest
of media/ never enter Hugo's content or static roots. Public builds require real
HTTPS media URLs; explicit local previews copy only the manifest's named files.

Design: an explicit article registry supplies each bundle's metadata, images,
and optional film. Keep historical media distinct from the current homepage
film. Resolve writeup links to published bundles and research links to GitHub.
Stage content and assets in a temporary directory, invoke Hugo, validate
local links/fragments and publication boundaries, then replace a fixed output
directory under .cache/site. A failed build preserves the previous output.
Serve only that output on loopback. No renderer dependencies or third-party
Python packages. Site governance: real product/narrow slice; test the contract.
"""
from __future__ import annotations

import argparse
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'
HUGO_VERSION = '0.154.2'
ARTICLES = {
    'milestone-1': {
        'title': 'Milestone 1: building the baseline',
        'description': 'The scene, the reference image, and the measurements behind the first '
                       'milestone, including the bugs that plausible images did not reveal.',
        'date': '2026-10-10', 'milestone': 1,
        'image': 'images/marbles-opening-10s.jpg',
        'images': ['marbles-opening-10s.jpg'],
        'film': 'baseline', 'film_title': 'The first milestone in motion',
    },
    'milestone-2': {
        'title': 'Milestone 2: reusing work with ReSTIR',
        'description': 'Keeping useful paths, giving the glass its own work, and measuring '
                       'what reuse buys. The connections to inference run through the pipeline.',
        'date': '2026-10-13', 'milestone': 2,
        'image': 'images/marbles-restir-pt-mnee-10s.jpg',
        'images': ['marbles-restir-pt-mnee-10s.jpg', 'marbles-path-vs-restir-pt-10s.jpg'],
        'film': 'comparison', 'film_title': 'The comparison in motion',
    },
    'milestone-3': {
        'title': 'Milestone 3: turning samples into an image',
        'description': 'Building and training a reconstruction model, running it inside the renderer, '
                       'and seeing what survives denoising and upscaling.',
        'date': '2026-10-16', 'milestone': 3,
        'image': 'images/marbles-denoised-20s.jpg',
        'images': ['marbles-denoised-20s.jpg', 'marbles-restir-pt-vs-denoised-10s.jpg'],
        'film': 'denoised_comparison', 'film_title': 'The reconstruction in motion',
    },
    'milestone-4': {
        'title': 'Milestone 4: making the frame fast',
        'description': 'Applying the optimizations shipped engines use once the image is right, '
                       'checking every stage against them, and retraining the network for the new input.',
        'date': '2026-10-19', 'milestone': 4,
        'image': 'images/marbles-m4-20s.jpg',
        'images': ['marbles-m4-20s.jpg', 'pipeline-m3.svg', 'pipeline-m4.svg', 'm4-overlap.svg',
                   'm4-shadow-rays.svg', 'm4-checkerboard.svg',
                   'm4-depth-of-field-near-marbles.jpg', 'm4-retrain.svg',
                   # each diagram drawn again for phones (render-image.html)
                   'pipeline-m3-narrow.svg', 'pipeline-m4-narrow.svg', 'm4-overlap-narrow.svg',
                   'm4-shadow-rays-narrow.svg', 'm4-checkerboard-narrow.svg', 'm4-retrain-narrow.svg'],
        'film': 'm3_vs_m4', 'film_title': 'Milestone 3 beside Milestone 4',
    },
}
IMAGE_KEYS = {'artwork', 'poster', 'baseline_poster', 'comparison_poster',
              'denoised_comparison_poster', 'm3_vs_m4_poster'}
MEDIA_KEYS = IMAGE_KEYS | {'film', 'film_small', 'film_original', 'brass',
                          'baseline', 'baseline_small', 'baseline_original',
                          'comparison', 'comparison_small', 'comparison_original',
                          'denoised_comparison', 'denoised_comparison_small',
                          'denoised_comparison_original',
                          'm3_vs_m4', 'm3_vs_m4_small', 'm3_vs_m4_original'}
PUBLIC_PAGES = {'index.html', 'about/index.html', 'writing/index.html', '404.html'} | {
    f'writing/{slug}/index.html' for slug in ARTICLES
}


def base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.netloc or
            parsed.query or parsed.fragment or not value.endswith('/')):
        raise ValueError('Base URL must be an HTTP(S) URL ending in /, without a query or fragment.')
    return value


def repository_file(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Missing or out-of-repository publication asset: {name}')
    return path


def copy_to(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage(stage_root: Path, local_media: bool) -> None:
    """Expose only the explicit public roots; never mount docs/ or media/."""
    for name in ('layouts', 'assets', 'static', 'content'):
        shutil.copytree(WEB / name, stage_root / name)
    shutil.copy2(WEB / 'hugo.toml', stage_root / 'hugo.toml')
    media = json.loads((WEB / 'data/media.json').read_text())
    if set(media) != MEDIA_KEYS:
        raise ValueError('Media manifest keys changed; review the publication allowlist in scripts/site.py.')
    missing = []
    for key, entry in media.items():
        remote = entry.get('url', '')
        if remote:
            parsed = urlsplit(remote)
            if parsed.scheme != 'https' or not parsed.netloc or parsed.username or parsed.password:
                raise ValueError(f'{key}: public media requires an HTTPS URL without credentials.')
        elif entry.get('optional'):
            # Optional originals/playback alternatives are shown only after explicit URL configuration.
            pass
        elif key in IMAGE_KEYS or local_media:
            source = repository_file(entry['local'])
            asset_name = f'images/{source.name}' if key in IMAGE_KEYS else f'media/{source.name}'
            copy_to(source, stage_root / 'static' / asset_name)
            entry['url'] = asset_name
        else:
            missing.append(key)
        # Local source/provenance stays in the manifest in Git, never in emitted HTML metadata.
        entry.pop('local', None)
    if missing:
        raise ValueError('Public media URLs are not configured for: ' + ', '.join(missing) +
                         '. Set web/data/media.json after hosting setup; use --local-media for review.')
    (stage_root / 'data').mkdir()
    (stage_root / 'data/media.json').write_text(json.dumps(media))
    for slug, article in ARTICLES.items():
        source = repository_file(f'docs/writeups/{slug}.md').read_text()
        heading, body = source.split('\n', 1)
        if heading != '# ' + article['title']:
            raise ValueError(f'{slug}: article title changed; update its metadata in scripts/site.py.')
        bundle = stage_root / 'content/writing' / slug
        bundle.mkdir(parents=True)
        metadata = {key: value for key, value in article.items() if key != 'images'}
        # JSON is valid Hugo frontmatter and avoids escaping YAML by hand.
        (bundle / 'index.md').write_text(json.dumps(metadata, indent=2) + '\n' + body)
        for name in article['images']:
            source_image = repository_file(f'docs/writeups/images/{name}')
            copy_to(source_image, bundle / 'images' / name)
            copy_to(source_image, stage_root / 'static/images' / name)


class Page(HTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.titles = self.main = self.h1 = 0
        self.feed(html)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                raise ValueError(f'Duplicate HTML id: {attrs["id"]}')
            self.ids.add(attrs['id'])
        self.titles += tag == 'title'
        self.main += tag == 'main'
        self.h1 += tag == 'h1'
        if tag == 'img' and 'alt' not in attrs:
            raise ValueError('Image missing alternative text')
        for key in ('href', 'src', 'poster'):
            if attrs.get(key):
                self.links.append(attrs[key])


def check_site(output: Path, base: str, local_media: bool) -> None:
    """Check real generated HTML and assets, including GitHub Pages path prefixes."""
    pages = {str(path.relative_to(output)): Page(path.read_text()) for path in output.rglob('*.html')}
    if set(pages) != PUBLIC_PAGES:
        raise ValueError(f'Unexpected public page set: {set(pages) ^ PUBLIC_PAGES}')
    origin = urlsplit(base)
    prefix = unquote(origin.path)
    for name, page in pages.items():
        if page.titles != 1 or page.main != 1 or page.h1 != 1:
            raise ValueError(f'{name}: expected one title, main, and h1')
        page_url = urljoin(base, name.removesuffix('index.html'))
        for link in page.links:
            parsed = urlsplit(urljoin(page_url, link))
            if parsed.scheme not in {'http', 'https'} or parsed.netloc != origin.netloc:
                continue
            path = unquote(parsed.path)
            if not path.startswith(prefix):
                raise ValueError(f'{name}: link escapes site base path: {link}')
            target = (output / path[len(prefix):]).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError(f'{name}: link escapes output: {link}')
            if target.is_dir():
                target /= 'index.html'
            if not target.is_file():
                raise ValueError(f'{name}: missing target {link}')
            if parsed.fragment and target.suffix == '.html':
                target_page = pages[str(target.relative_to(output.resolve()))]
                if unquote(parsed.fragment) not in target_page.ids:
                    raise ValueError(f'{name}: missing fragment {link}')
    # CSS URLs are not seen by the HTML parser.
    for css in output.glob('*.css'):
        for raw in re.findall(r'url\([\"\']?([^\)\"\']+)', css.read_text()):
            if not raw.startswith('data:') and not (css.parent / raw).is_file():
                raise ValueError(f'Missing CSS asset: {raw}')
    for path in output.rglob('*'):
        if path.is_file() and path.suffix.lower() not in {'.html', '.xml', '.txt', '.css', '.svg', '.png', '.jpg', '.woff2', '.mp4'}:
            raise ValueError(f'Unexpected published file: {path.relative_to(output)}')
    if not local_media and (output / 'media').exists():
        raise ValueError('A public build must not contain local media files.')


def check_hugo_version(version: str) -> None:
    # Official releases include a source commit; Homebrew omits it. Accept both
    # forms of the exact release, but never a prerelease or another patch.
    pattern = r'^hugo v' + re.escape(HUGO_VERSION) + r'(?:-[0-9a-f]{40})?(?:\+|\s)'
    if not re.search(pattern, version):
        raise ValueError(f'Use Hugo {HUGO_VERSION}; found {version.strip()}')


def build(base: str, local_media: bool) -> Path:
    base_url(base)
    result = subprocess.run(['hugo', 'version'], check=True, capture_output=True, text=True)
    check_hugo_version(result.stdout)
    destination = ROOT / '.cache/site' / ('preview' if local_media else 'public')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='serenity-site-', dir=destination.parent) as temp:
        scratch = Path(temp)
        project, output = scratch / 'project', scratch / 'output'
        project.mkdir()
        stage(project, local_media)
        subprocess.run(['hugo', '--source', str(project), '--destination', str(output),
                        '--baseURL', base, '--panicOnWarning'], check=True)
        check_site(output, base, local_media)
        previous = scratch / 'previous'
        if destination.exists():
            destination.rename(previous)
        try:
            output.rename(destination)
        except OSError:
            if previous.exists():
                previous.rename(destination)
            raise
    print(f'Built and checked {len(PUBLIC_PAGES) - 1} content pages + 404: {destination}', flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'serve'))
    parser.add_argument('--local-media', action='store_true', help='Use selected ignored media for local review only')
    parser.add_argument('--base-url', default='https://serenity.plainsight-systems.com/')
    parser.add_argument('--port', type=int, default=1313)
    args = parser.parse_args()
    try:
        base = f'http://127.0.0.1:{args.port}/' if args.command == 'serve' else args.base_url
        output = build(base, args.local_media)
        if args.command == 'serve':
            print(f'Preview: {base} (Ctrl-C to stop; rerun after editing)', flush=True)
            handler = partial(SimpleHTTPRequestHandler, directory=str(output))
            with ThreadingHTTPServer(('127.0.0.1', args.port), handler) as server:
                server.serve_forever()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Site build failed: {error}\n')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
