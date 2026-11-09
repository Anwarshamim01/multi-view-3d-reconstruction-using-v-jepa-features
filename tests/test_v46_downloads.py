"""Exercise real HTTP downloads/extraction without downloading a research dataset."""
from contextlib import contextmanager
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import zipfile

import numpy as np
from PIL import Image
import pytest

from kaggle_v46 import data
from kaggle_v46 import co3d_download as download
from kaggle_v46.config import default_config


@contextmanager
def server(files):
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            content=files[self.path];start=0
            header=self.headers.get('Range')
            if header:start=int(header.split('=')[1].split('-')[0])
            requests.append((self.path,start))
            self.send_response(206 if header else 200)
            self.send_header('Content-Length',str(len(content)-start))
            if header:self.send_header('Content-Range',f'bytes {start}-{len(content)-1}/{len(content)}')
            self.end_headers();self.wfile.write(content[start:])
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
    try:yield f'http://127.0.0.1:{http.server_port}',requests
    finally:http.shutdown();http.server_close();worker.join()


def image_bytes(array):
    buffer=io.BytesIO();Image.fromarray(array).save(buffer,format='PNG');return buffer.getvalue()


def archive_bytes(entries):
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,content in entries.items():z.writestr(name,content)
    return buffer.getvalue()


def test_default_data_and_assets_use_kaggle_temp():
    for dataset,path in [('co3d','/kaggle/temp/co3d_data'),('tum','/kaggle/temp/datasets/tum_rgbd')]:
        c=default_config(dataset)
        assert c['allow_download'] is True
        assert str(data.locate_root(c))==path
        assert c['asset_root'].startswith('/kaggle/temp/')


def test_no_attachment_download_resume_extract_reuse_and_repair(tmp_path,monkeypatch):
    c=default_config('co3d')
    c.update(image_size=32,window_views=3,frame_stride=1,window_step=1,
        splits={s:['apple/'+s] for s in ['train','val','test']})
    rows=[];images={}
    for seq in ['train','val','test']:
        for index in range(3):
            prefix=f'apple/{seq}/{index}'
            rgb,depth,mask=prefix+'.png',prefix+'_depth.png',prefix+'_mask.png'
            images[rgb]=image_bytes(np.full((32,32,3),100,np.uint8))
            images[depth]=image_bytes(np.full((32,32),1.5,np.float16).view(np.uint16))
            images[mask]=image_bytes(np.full((32,32),255,np.uint8))
            rows.append(dict(sequence_name=seq,frame_number=index,image=dict(path=rgb,size=[32,32]),
                depth=dict(path=depth,scale_adjustment=1.),mask=dict(path=mask),
                viewpoint=dict(focal_length=[2.,2.],principal_point=[0.,0.],
                    R=np.eye(3).tolist(),T=[index*.01,0.,0.])))
    blobs={'/meta.zip':archive_bytes({'apple/frame_annotations.jgz':gzip.compress(json.dumps(rows).encode())}),
           '/data.zip':archive_bytes(images)}
    cache=tmp_path/'temp/co3d_data'
    monkeypatch.setattr(data,'CO3D_DEFAULT_ROOT',cache)
    with server(blobs) as (url,requests):
        monkeypatch.setitem(download.ARCHIVES,'apple',[
            dict(url=url+name,sha256=hashlib.sha256(blob).hexdigest()) for name,blob in blobs.items()])
        # Start with a real interrupted HTTP transfer: server must receive and honor Range.
        partial=cache/'_downloads/meta.zip.part';partial.parent.mkdir(parents=True)
        half=len(blobs['/meta.zip'])//2;partial.write_bytes(blobs['/meta.zip'][:half])
        manifest=data.prepare_manifest(c,tmp_path/'manifest.json')
        assert requests[0]==('/meta.zip',half)
        assert len(requests)==2
        assert all(len(windows)==1 for windows in manifest['splits'].values())
        assert len(manifest['frames'])==9
        assert all(Path(frame['rgb']).is_relative_to(cache) for frame in manifest['frames'].values())
        assert not list((cache/'_downloads').glob('*.zip'))
        # A second pass reuses all complete data, without issuing HTTP requests.
        second=data.prepare_manifest(c,tmp_path/'second.json')
        assert second['fingerprint']==manifest['fingerprint'] and len(requests)==2
        # An existing annotation file cannot hide a missing depth frame.
        Path(next(iter(manifest['frames'].values()))['depth']).unlink()
        data.prepare_manifest(c,tmp_path/'repaired.json')
        assert len(requests)==4
        assert data.co3d_category_ready(cache,'apple',{'apple/train','apple/val','apple/test'})


def test_bad_checksum_never_becomes_complete_download(tmp_path):
    with server({'/data.zip':b'bad payload'}) as (url,requests):
        target=tmp_path/'data.zip'
        with pytest.raises(RuntimeError,match='Download failed'):
            download.download_file(url+'/data.zip',target,'0'*64,attempts=1)
        assert not target.exists()


def test_zip_cannot_escape_category(tmp_path):
    archive=tmp_path/'bad.zip';archive.write_bytes(archive_bytes({'../outside':b'no'}))
    with pytest.raises(ValueError,match='Unexpected CO3D'):
        download.extract_category(archive,tmp_path/'data','apple')
    assert not (tmp_path/'outside').exists()


def test_partial_tum_sequence_is_not_reused(tmp_path):
    root=tmp_path/'sequence'
    (root/'rgb').mkdir(parents=True);(root/'depth').mkdir()
    (root/'groundtruth.txt').write_text('0 0 0 0 0 0 0 1\n')
    (root/'rgb.txt').write_text('0 rgb/0.png\n');(root/'depth.txt').write_text('0 depth/0.png\n')
    (root/'rgb/0.png').write_bytes(b'image')
    assert not data.tum_sequence_ready(root)
    (root/'depth/0.png').write_bytes(b'depth')
    assert data.tum_sequence_ready(root)
