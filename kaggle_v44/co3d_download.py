"""Official CO3D subset archives for the unchanged twelve-sequence experiment.

Links and checksums copied from facebookresearch/co3d at the pinned revision below.
Downloads are sequential, resumable, checked, and stored under /kaggle/temp.
"""
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def download_file(url,path,expected_sha256=None,attempts=4):
    """Atomic, retried HTTP download; resume a partial transfer when Range is supported."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and (expected_sha256 is None or sha256(path)==expected_sha256):return path
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(attempts):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset and expected_sha256 and sha256(partial)==expected_sha256:
                partial.replace(path);return path
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
            print(f'Downloading {url} (attempt {attempt+1}/{attempts}, cached {offset/2**20:.1f} MiB)',flush=True)
            with urllib.request.urlopen(request,timeout=90) as response:
                resumed=offset>0 and response.status==206
                if resumed and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned an inconsistent resume range')
                if not resumed:offset=0
                length=response.headers.get('Content-Length')
                expected_size=offset+int(length) if length is not None else None
                received=offset;last=time.monotonic()
                with partial.open('ab' if resumed else 'wb') as stream:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        stream.write(block);received+=len(block)
                        if time.monotonic()-last>=10:
                            print(f'  {path.name}: {received/2**20:.1f} MiB downloaded',flush=True);last=time.monotonic()
                if expected_size is not None and received!=expected_size:
                    raise OSError(f'Incomplete transfer: {received}/{expected_size} bytes')
            if expected_sha256 and sha256(partial)!=expected_sha256:
                partial.unlink()  # Only this downloader's corrupt temporary transfer.
                raise ValueError('Official CO3D SHA256 verification failed: '+path.name)
            partial.replace(path)
            print('Download complete:',path,flush=True)
            return path
        except (OSError,ValueError,urllib.error.URLError,http.client.HTTPException) as error:
            if isinstance(error,urllib.error.HTTPError) and error.code==416 and partial.exists():
                partial.unlink()  # A stale range cannot be resumed; retry from byte zero.
            if attempt+1==attempts:
                raise RuntimeError(f'Download failed after {attempts} attempts: {url}. Enable Kaggle Internet and rerun; partial downloads are retained.') from error
            print('Download interrupted; retrying:',str(error),flush=True)
            time.sleep(min(2**attempt,8))


def extract_category(archive,root,category):
    """Extract only category members and reject traversal/symlinks before writing."""
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            member=Path(info.filename)
            if member.is_absolute() or '..' in member.parts or not member.parts or member.parts[0]!=category:
                raise ValueError('Unexpected CO3D archive member: '+info.filename)
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('CO3D symlink entry rejected')
            target=root/member
            if not target.resolve().is_relative_to(root/category):raise ValueError('CO3D target escapes its category')
        z.extractall(root)


def download_category(root,category):
    if category not in ARCHIVES:
        raise ValueError(f'No automatic archive manifest for custom CO3D category {category}; configure an official source before changing the experiment split')
    root=Path(root);downloads=root/'_downloads';downloads.mkdir(parents=True,exist_ok=True)
    receipts=[]
    for record in ARCHIVES[category]:
        archive=downloads/record['url'].rsplit('/',1)[-1]
        download_file(record['url'],archive,record['sha256'])
        print('Extracting',archive.name,'to',root,flush=True)
        extract_category(archive,root,category)
        receipts.append(record)
        archive.unlink()  # Release space only after this verified archive was extracted.
    receipt=root/category/'v44_download_provenance.json'
    receipt.write_text(json.dumps(dict(upstream_revision=CO3D_REVISION,
        subset='official_singlesequence_category_archives',archives=receipts),indent=2))

CO3D_REVISION = 'eb51d7583c56ff23dc918d9deafee50f4d8178c3'
ARCHIVES = {'apple': [{'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/apple_000_singlesequence.zip', 'sha256': 'e461c00b6022592f0a9eb23c9c24886902db6382e04fefab2a17c782198777ba'}, {'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/apple_001_singlesequence.zip', 'sha256': 'dd58ede7193a9801c92143a8a65abb842d188011b0568680accd1b281f9bc885'}], 'ball': [{'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/ball_000_singlesequence.zip', 'sha256': '7a9bb596b5286809dc701bb8a5541d9c2d577d2d2975b22b98144aa34eab77b2'}, {'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/ball_001_singlesequence.zip', 'sha256': '8cd6861f2d0d037dfc3ccf8eb60c39ae76ada2493d28f99eccdd9e29bf3ffc47'}], 'bench': [{'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/bench_000_singlesequence.zip', 'sha256': 'cffb3c2a2510c4bcf9e52d24acb108ee351133599ca7d192172f64862c685dbf'}, {'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/bench_001_singlesequence.zip', 'sha256': '2d72c796a121f88bea043af330a4b91b0936e15057ba149d042a31741a92d19f'}], 'bowl': [{'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/bowl_000_singlesequence.zip', 'sha256': '9b8ca3a579f3b60920ad48f50d72036f01d2c696e605aa08767bbbb5ce5eb7e6'}, {'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/bowl_001_singlesequence.zip', 'sha256': '491f003f92d29ecbaf34e3cbbaf7e46bb387866c117c64c4f465ab279666910f'}], 'broccoli': [{'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/broccoli_000_singlesequence.zip', 'sha256': 'a3abb5637ddc1bd305b7e154bd027c45a377105b18fc28acfe07890e32553282'}, {'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/broccoli_001_singlesequence.zip', 'sha256': 'daf794919ec7d8a21c656c59ac56f8215cb55bca51d59b8fe4531e75a6577102'}], 'cake': [{'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/cake_000_singlesequence.zip', 'sha256': '34de5d1a76995e99d045cd1d653aebe21b23acc2d0ee9597487b4f6ba337dd84'}, {'url': 'https://dl.fbaipublicfiles.com/co3dv2_231130/cake_001_singlesequence.zip', 'sha256': '8bab2d0dcb4027782b1c0e55723d1269bfb84b629b6539b4a03c510f95506a87'}]}
