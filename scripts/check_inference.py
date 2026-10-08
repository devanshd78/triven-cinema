#!/usr/bin/env python3
"""Validate inference configuration through the CPU worker without rendering video."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'services' / 'api')]


def main() -> int:
    from app.core.config import settings
    from inference.providers.router import get_video_provider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Check every enabled production rendering recipe')
    args = parser.parse_args()
    provider = get_video_provider(settings.video_provider, model='ltx-2.5')
    recipes = [('1080p standard', dict(render_mode='dfr', decoder='diffusion', realism_profile='standard'))]
    if args.all:
        recipes += [('preview', dict(render_mode='distilled', decoder=settings.default_decoder)),
                    ('Real Skin', dict(render_mode='dfr', decoder='diffusion', realism_profile='real_skin'))]
        if settings.element_ingredients_enabled:
            recipes.append(('Character identity', dict(render_mode='dfr', decoder='diffusion', realism_profile='identity_max', element_reference_required=True)))
        if settings.factory_audio_retake_enabled:
            recipes.append(('audio repair', dict(operation='retake_audio', decoder='diffusion')))
    failures = 0
    for label, recipe in recipes:
        try:
            manifest = provider.preflight(**recipe, force_refresh=True)
            if not manifest.get('ready'):
                raise RuntimeError('; '.join(manifest.get('errors') or ['Readiness could not be verified.']))
            print(f"[OK] {label}: worker protocol {manifest.get('protocol_version')}, LTX {manifest.get('ltx_repo_ref')}, GPU {manifest.get('gpu')}")
        except Exception as exc:
            failures += 1
            print(f'[FAIL] {label}: {exc}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
