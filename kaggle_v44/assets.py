"""Pinned encoder source and canonical checkpoint acquisition."""
from pathlib import Path
import urllib.request
import zipfile
from .data import file_hash,atomic_json


def prepare_assets(c):
    root=Path(c['asset_root']);root.mkdir(parents=True,exist_ok=True)
    revision=c['upstream_revision']
    repo=root/('vjepa2-'+revision)
    if not (repo/'app/vjepa_2_1/models/vision_transformer.py').is_file():
        if not c['allow_download']:
            raise FileNotFoundError('Pinned encoder source not found: '+str(repo))
        archive=root/('source-'+revision+'.zip')
        urllib.request.urlretrieve('https://codeload.github.com/facebookresearch/vjepa2/zip/'+revision,archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if '..' in Path(name).parts or Path(name).is_absolute():
                    raise ValueError('Unsafe source archive entry')
            z.extractall(root)
        archive.unlink()
    checkpoint=root/c['encoder_checkpoint']
    if not checkpoint.exists():
        if not c['allow_download']:
            raise FileNotFoundError(str(checkpoint))
        temporary=checkpoint.with_suffix('.downloading')
        print('Downloading canonical encoder checkpoint',flush=True)
        urllib.request.urlretrieve(c['encoder_url'],temporary)
        temporary.replace(checkpoint)
    digest=file_hash(checkpoint)
    if c.get('encoder_sha256') and digest!=c['encoder_sha256']:
        raise ValueError('Encoder SHA256 mismatch')
    manifest=root/'encoder_provenance.json'
    if not manifest.exists():
        atomic_json(manifest,dict(revision=revision,url=c['encoder_url'],sha256=digest,
                                  checkpoint_key='ema_encoder'))
    return repo,checkpoint
