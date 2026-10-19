"""Publication-boundary and navigation tests against real Hugo output.

Contract: a Pages subpath must work, missing media must fail before publication,
and accidental private pages or broken anchors must be rejected. Use temporary
staging and fixture HTTPS URLs; never upload, fetch media, or mutate repo sources.
"""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('serenity_site', Path(__file__).resolve().parents[1] / 'scripts/site.py')
site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


class SiteTests(unittest.TestCase):
    def replaced(self, page, old, new):
        """`page`'s text with `old` replaced by `new`; `old` must be there, or the
        test would check the page unchanged and fail as "did not raise"."""
        text = page.read_text()
        self.assertIn(old, text, f'{page.name} no longer holds {old!r}')
        return text.replace(old, new)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.web = self.root / 'web'
        shutil.copytree(site.WEB, self.web)
        self.patcher = patch.object(site, 'WEB', self.web)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_pinned_hugo_accepts_release_packaging_only(self):
        for version in (
            'hugo v0.154.2+extended+withdeploy darwin/arm64',
            'hugo v0.154.2-f66d0944461bf32c4e69588bc3e093f14e4e149d+extended linux/amd64',
            'hugo v0.154.2 linux/amd64',
        ):
            with self.subTest(version=version):
                site.check_hugo_version(version)
        for version in ('hugo v0.154.20+extended', 'hugo v0.154.3+extended',
                        'hugo v0.154.2-rc.1+extended', 'hugo v0.154.2-dev+extended'):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, 'Use Hugo'):
                    site.check_hugo_version(version)

    def configure_media(self):
        manifest = self.web / 'data/media.json'
        data = json.loads(manifest.read_text())
        for name, entry in data.items():
            if name not in site.IMAGE_KEYS:
                entry['url'] = 'https://media.example.invalid/' + Path(entry['local']).name
        manifest.write_text(json.dumps(data))

    def render(self):
        self.configure_media()
        project = self.root / 'project'
        project.mkdir()
        site.stage(project, False)
        output = self.root / 'output'
        base = 'https://example.invalid/serenity/'
        subprocess.run(['hugo', '--source', str(project), '--destination', str(output),
                        '--baseURL', base, '--panicOnWarning'], check=True, capture_output=True)
        return project, output, base

    def test_public_build_requires_hosted_media(self):
        # Exercise missing configuration independently of the live manifest.
        manifest = self.web / 'data/media.json'
        data = json.loads(manifest.read_text())
        for name, entry in data.items():
            # Keep hosted posters configured: their ignored local originals
            # need not exist in CI. Exercise missing required film URLs only.
            if name not in site.IMAGE_KEYS:
                entry['url'] = ''
        manifest.write_text(json.dumps(data))
        project = self.root / 'project'
        project.mkdir()
        with self.assertRaisesRegex(ValueError, 'URLs are not configured for: film, brass'):
            site.stage(project, False)

    def test_real_pages_work_under_project_subpath(self):
        project, output, base = self.render()
        site.check_site(output, base, False)
        self.assertFalse((output / 'media').exists())
        self.assertFalse((project / 'content/process').exists())
        self.assertFalse((project / 'content/research').exists())
        self.assertEqual(set((project / 'content/writing').rglob('*.md')),
                         {project / 'content/writing/_index.md',
                          project / 'content/writing/milestone-1/index.md',
                          project / 'content/writing/milestone-2/index.md',
                          project / 'content/writing/milestone-3/index.md',
                          project / 'content/writing/milestone-4/index.md'})
        html = (output / 'index.html').read_text()
        self.assertIn('https://media.example.invalid/', html)
        self.assertIn('Smaller playback file', html)
        self.assertNotIn('media/2026-', html)
        article = (output / 'writing/milestone-1/index.html').read_text()
        self.assertIn('role="region"', article)
        self.assertIn('scope="col"', article)

    def test_milestones_keep_their_media_and_repository_links(self):
        _, output, _ = self.render()
        first = (output / 'writing/milestone-1/index.html').read_text()
        second = (output / 'writing/milestone-2/index.html').read_text()
        home = (output / 'index.html').read_text()
        index = (output / 'writing/index.html').read_text()
        self.assertIn('2026-10-10-marbles-opening-v3-1080p30-16mbps.mp4', first)
        self.assertIn('2026-10-19-marbles-opening-m4-denoised-1080p30-16mbps.mp4', home)
        self.assertIn('href="/serenity/writing/milestone-4/"', home)
        self.assertNotIn('2026-10-15-marbles-opening-restir-pt-denoised-1080p30-16mbps.mp4', home)
        self.assertNotIn('2026-10-10-marbles-opening-v3-1080p30-16mbps.mp4', home)
        self.assertIn('2026-10-13-marbles-opening-path-vs-restir-pt-1080p30-16mbps.mp4', second)
        self.assertIn('width="1920" height="540"', second)
        self.assertIn('href="/serenity/writing/milestone-1/"', second)
        self.assertIn('https://github.com/plainsight-systems/serenity/blob/main/docs/research/'
                      '2026-10-12-mnee.md#equal-frame-time-one-displayed-frame-each', second)
        self.assertIn('https://example.invalid/serenity/images/marbles-restir-pt-mnee-10s.jpg', second)
        self.assertIn('https://example.invalid/serenity/images/marbles-opening-10s.jpg', first)
        third = (output / 'writing/milestone-3/index.html').read_text()
        self.assertIn('2026-10-15-marbles-opening-restir-pt-vs-denoised-1080p30-16mbps.mp4', third)
        self.assertIn('width="1920" height="540"', third)
        self.assertIn('href="/serenity/writing/milestone-2/"', third)
        self.assertIn('https://example.invalid/serenity/images/marbles-denoised-20s.jpg', third)
        self.assertIn('2026-10-15-denoiser-training.md', third)
        fourth = (output / 'writing/milestone-4/index.html').read_text()
        self.assertIn('2026-10-19-marbles-opening-m3-vs-m4-1080p30-16mbps.mp4', fourth)
        self.assertIn('width="1920" height="540"', fourth)
        self.assertIn('href="/serenity/writing/milestone-3/"', fourth)
        self.assertIn('https://example.invalid/serenity/images/marbles-m4-20s.jpg', fourth)
        self.assertNotIn('2026-10-19-marbles-opening-m3-vs-m4-1080p30-16mbps.mp4', third)
        # Diagrams are published beside the article and wrapped for phone-width scrolling.
        for name in ('pipeline-m3', 'pipeline-m4', 'm4-overlap', 'm4-shadow-rays',
                     'm4-checkerboard', 'm4-retrain'):
            self.assertTrue((output / f'writing/milestone-4/images/{name}.svg').is_file())
            self.assertIn(f'role="region" aria-label="Diagram" tabindex="0"><img class="diagram" '
                          f'src="images/{name}.svg"', fourth)
        self.assertNotIn('class="diagram"', first + second + third)
        self.assertLess(index.index('writing/milestone-4/'), index.index('writing/milestone-3/'))
        self.assertLess(index.index('writing/milestone-3/'), index.index('writing/milestone-2/'))
        self.assertLess(index.index('writing/milestone-2/'), index.index('writing/milestone-1/'))

    def test_feed_is_complete_public_metadata_even_in_preview(self):
        project, output, base = self.render()
        # More than a typical feed page, plus private/future entries. These are
        # preview-only fixtures, never publication-registry changes.
        for number in range(25):
            target = project / 'content/writing' / f'fixture-{number}.md'
            target.write_text(json.dumps({'title': f'Fixture {number} & <GPU>',
                                         'date': '2020-01-01',
                                         'description': 'A & B < C'}) + '\nPRIVATE BODY')
        for slug, metadata in [('draft', {'draft': True, 'date': '2020-01-01'}),
                               ('future', {'date': '2099-01-01'}),
                               ('scheduled', {'date': '2020-01-01', 'publishDate': '2099-01-01'})]:
            (project / f'content/writing/{slug}.md').write_text(
                json.dumps({'title': slug, **metadata}) + '\nPRIVATE BODY')
        subprocess.run(['hugo', '--source', str(project), '--destination', str(output),
                        '--baseURL', base, '--buildDrafts', '--buildFuture',
                        '--panicOnWarning'], check=True, capture_output=True)
        xml = (output / 'writing/index.xml').read_text()
        entries = ET.fromstring(xml).findall('./channel/item')
        self.assertEqual(len(entries), 29)
        self.assertEqual(entries[0].findtext('title'), site.ARTICLES['milestone-4']['title'])
        self.assertIn('Fixture 0 & <GPU>', [item.findtext('title') for item in entries])
        self.assertNotIn('PRIVATE BODY', xml)
        self.assertNotIn('/writing/draft/', xml)
        self.assertNotIn('/writing/future/', xml)
        self.assertNotIn('/writing/scheduled/', xml)
        self.assertFalse((output / 'about/index.xml').exists())
        for item in entries:
            self.assertTrue(item.findtext('link').startswith(base + 'writing/'))
            self.assertEqual(item.findtext('guid'), item.findtext('link'))
        self.assertIn(base + 'writing/index.xml', (output / 'index.html').read_text())

    def test_unselected_article_cannot_be_published(self):
        _, output, base = self.render()
        (output / 'private.html').write_text('<h1>Unselected</h1>')
        with self.assertRaisesRegex(ValueError, 'Unexpected public page set'):
            site.check_site(output, base, False)

    def test_checker_rejects_missing_fragments(self):
        _, output, base = self.render()
        page = output / 'index.html'
        page.write_text(self.replaced(page, 'href="#film"', 'href="#missing"'))
        with self.assertRaisesRegex(ValueError, 'missing fragment'):
            site.check_site(output, base, False)

    def test_checker_rejects_subpath_escape(self):
        _, output, base = self.render()
        page = output / 'index.html'
        page.write_text(self.replaced(page, 'href="/serenity/writing/"', 'href="/writing/"'))
        with self.assertRaisesRegex(ValueError, 'escapes site base path'):
            site.check_site(output, base, False)

    def test_checker_rejects_missing_assets(self):
        _, output, base = self.render()
        (output / 'fonts/PlexSerif-Regular.woff2').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing CSS asset'):
            site.check_site(output, base, False)

    def test_local_source_cannot_escape_repo(self):
        with self.assertRaisesRegex(ValueError, 'out-of-repository'):
            site.repository_file('../cradle/README.md')


if __name__ == '__main__':
    unittest.main()
