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
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('serenity_site', Path(__file__).resolve().parents[1] / 'scripts/site.py')
site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


class SiteTests(unittest.TestCase):
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
        for name in ('film', 'brass', 'film_small', 'film_original'):
            data[name]['url'] = 'https://media.example.invalid/' + Path(data[name]['local']).name
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
        for entry in data.values():
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
                         {project / 'content/writing/_index.md', project / 'content/writing/milestone-1/index.md'})
        html = (output / 'index.html').read_text()
        self.assertIn('https://media.example.invalid/', html)
        self.assertIn('Smaller playback file', html)
        self.assertNotIn('media/2026-', html)
        article = (output / 'writing/milestone-1/index.html').read_text()
        self.assertIn('role="region"', article)
        self.assertIn('scope="col"', article)

    def test_unselected_article_cannot_be_published(self):
        _, output, base = self.render()
        (output / 'private.html').write_text('<h1>Unselected</h1>')
        with self.assertRaisesRegex(ValueError, 'Unexpected public page set'):
            site.check_site(output, base, False)

    def test_checker_rejects_missing_fragments(self):
        _, output, base = self.render()
        page = output / 'index.html'
        page.write_text(page.read_text().replace('href="#film"', 'href="#missing"'))
        with self.assertRaisesRegex(ValueError, 'missing fragment'):
            site.check_site(output, base, False)

    def test_checker_rejects_subpath_escape(self):
        _, output, base = self.render()
        page = output / 'index.html'
        page.write_text(page.read_text().replace('href="/serenity/writing/"', 'href="/writing/"'))
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
