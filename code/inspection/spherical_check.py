"""CPU ERP inspection with fresh output folders beside the input image."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageOps

VERSION = '0.2.0'
SIX_VIEWS = [('front',0,0),('right',90,0),('back_seam',180,0),('left',270,0),('zenith',0,90),('nadir',0,-90)]
EIGHT_VIEWS = [(f'{i:02d}_yaw_{yaw:03d}',yaw,0) for i,yaw in enumerate(range(0,360,45))]+[('08_zenith',0,90),('09_nadir',0,-90)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def logical_path(path):
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def sample_erp(image,x,y):
    image=np.asarray(image)
    h,w=image.shape[:2]
    x=np.mod(x,w)
    y=np.clip(y,0,h-1)
    x0=np.floor(x).astype(np.int64)
    y0=np.floor(y).astype(np.int64)
    x1=(x0+1)%w
    y1=np.minimum(y0+1,h-1)
    ax=(x-x0)[...,None]
    ay=(y-y0)[...,None]
    return (image[y0,x0]*(1-ax)+image[y0,x1]*ax)*(1-ay)+(image[y1,x0]*(1-ax)+image[y1,x1]*ax)*ay


def perspective(image,yaw_deg=0,pitch_deg=0,fov_deg=90,size=512):
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3 or image.shape[1]!=2*image.shape[0]:
        raise ValueError('Expected 2:1 RGB equirectangular image')
    if not math.isfinite(fov_deg) or not 0<fov_deg<180 or not isinstance(size,int) or size<1:
        raise ValueError('Invalid field of view or size')
    h,w=image.shape[:2]
    yaw,pitch=np.deg2rad([yaw_deg,pitch_deg])
    forward=np.array([np.sin(yaw)*np.cos(pitch),np.sin(pitch),np.cos(yaw)*np.cos(pitch)])
    right=np.array([np.cos(yaw),0,-np.sin(yaw)])
    up=np.cross(forward,right)
    axis=(2*(np.arange(size)+.5)/size-1)*np.tan(np.deg2rad(fov_deg)/2)
    xx,yy=np.meshgrid(axis,-axis)
    rays=forward+xx[...,None]*right+yy[...,None]*up
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    lon=np.arctan2(rays[...,0],rays[...,2])
    lat=np.arcsin(np.clip(rays[...,1],-1,1))
    return sample_erp(image,(lon/(2*np.pi)+.5)*w-.5,(.5-lat/np.pi)*h-.5)


def fresh_output(image):
    base=image.parent/(image.stem+'_spherical_check')
    try:
        base.mkdir()
        return base
    except FileExistsError:
        pass
    while True:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        candidate=base.with_name(base.name+'_'+stamp+'_'+uuid.uuid4().hex[:8])
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue


def as_image(array):
    return Image.fromarray(np.clip(np.rint(array),0,255).astype(np.uint8))


def check_panorama(image,source=None,size=512,fov=90,layout='eight'):
    if layout not in ('eight','six'):
        raise ValueError('Layout must be eight or six')
    views=EIGHT_VIEWS if layout=='eight' else SIX_VIEWS
    columns=4 if layout=='eight' else 3
    rows=math.ceil(len(views)/columns)
    if not isinstance(size,int) or isinstance(size,bool) or not 16<=size<=2048:
        raise ValueError('View size must be an integer in [16,2048]')
    if not math.isfinite(fov) or not 0<fov<180:
        raise ValueError('Field of view must be finite and between 0 and 180 degrees')
    image=logical_path(image)
    if not image.is_file():
        raise FileNotFoundError(image)
    input_hash=digest(image)
    with Image.open(image) as im:
        im.load()
        if im.mode!='RGB' or im.width!=2*im.height:
            raise ValueError('Input must already be a 2:1 RGB ERP; no silent conversion/resize')
        erp=np.array(im)
    source_image=None
    source_info=None
    if source is not None:
        source=logical_path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash=digest(source)
        with Image.open(source) as im:
            im.load()
            source_image=im.convert('RGB').copy()
            source_info={'path':str(source),'sha256':source_hash,'original_mode':im.mode,'size':list(im.size)}
    out=fresh_output(image)
    manifest={
        'tool':'panorama-spherical-check','version':VERSION,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'input':{'path':str(image),'resolved_path':str(image.resolve()),'sha256':input_hash,'mode':'RGB','size':[int(erp.shape[1]),int(erp.shape[0])]},
        'source':source_info,'output_dir':str(out),'cpu_only':True,
        'projection_status':'rendering','review_status':'pending_manual_review','review_file':'review.md',
        'config':{'layout':layout,'view_size':size,'fov_degrees':fov,'fov_horizontal_degrees':fov,'fov_vertical_degrees':fov,'yaw0':'ERP center','yaw_positive':'turn right','yaw180':'ERP wrap seam','pitch_positive':'up','same_camera_center':True},
        'views':[{'name':n,'yaw':y,'pitch':p} for n,y,p in views],
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(size*columns,(size+28)*rows),'#202020')
    draw=ImageDraw.Draw(sheet)
    for i,(name,yaw,pitch) in enumerate(views):
        view=as_image(perspective(erp,yaw,pitch,fov,size))
        view.save(out/(name+'.png'))
        x=(i%columns)*size; y=(i//columns)*(size+28)
        sheet.paste(view,(x,y+28))
        draw.text((x+5,y+7),f'{name} | {fov:g}deg',fill='white')
    sheet.save(out/'spherical_views.jpg',quality=92)
    preview=ImageOps.contain(Image.fromarray(erp),(1200,600))
    preview.save(out/'erp_preview.jpg',quality=92)
    if source_image is not None:
        compare=Image.new('RGB',(size*2,size+30),'#202020')
        d=ImageDraw.Draw(compare)
        thumb=ImageOps.contain(source_image,(size,size))
        compare.paste(thumb,((size-thumb.width)//2,30+(size-thumb.height)//2))
        compare.paste(as_image(perspective(erp,0,0,80,size)),(size,30))
        d.text((5,7),'Source (not registered)',fill='white')
        d.text((size+5,7),'Front | nominal 80deg',fill='white')
        compare.save(out/'source_vs_front.jpg',quality=92)
        flat=Image.new('RGB',(1200,1000),'#202020')
        fd=ImageDraw.Draw(flat)
        thumb=ImageOps.contain(source_image,(500,350))
        flat.paste(thumb,((1200-thumb.width)//2,35))
        flat.paste(Image.fromarray(erp).resize((1200,600)),(0,400))
        fd.text((10,10),'Original source',fill='white')
        fd.text((10,380),'Generated ERP (not a registered comparison)',fill='white')
        flat.save(out/'source_vs_erp.jpg',quality=92)
        manifest['source_comparison_fov_degrees']=80
    # These values are descriptive only, not pass/fail quality metrics.
    edge=np.abs(erp[:,0].astype(float)-erp[:,-1].astype(float)).mean()
    adjacent=np.abs(np.diff(erp.astype(np.float32),axis=1)).mean(axis=(0,2))
    manifest['diagnostics']={'wrap_edge_rgb_mae':float(edge),'median_internal_horizontal_edge_rgb_mae':float(np.median(adjacent)),'quality_threshold':None}
    review='''# Spherical inspection review

Status: PENDING MANUAL REVIEW

## Input and outputs
See manifest.json for original image hash, source, projection convention and file hashes.

## Observations (fill only after viewing the saved images)
- Source/front retention: pending
- Scene and style continuity: pending
- Back-wrap seam: pending
- Ceiling/sky: pending
- Floor/ground: pending
- Invented/repeated objects or inconsistent room layout: pending

## Verdict
Pending: accept / needs review / reject / inconclusive, with reasons.

## Limits
This is same-center 2D inspection, not a 3D reconstruction validation or proof
that the input is a physically valid ERP. Smooth seams do not certify semantic
or spatial consistency. Source comparison is not pixel-registered. No automatic
quality acceptance threshold is applied.
'''
    (out/'review.md').write_text(review,encoding='utf-8')
    if digest(image)!=input_hash or (source_info and digest(source)!=source_info['sha256']):
        raise RuntimeError('Input/source changed during inspection; outputs left for diagnosis')
    outputs=[]
    for p in sorted(out.iterdir()):
        if p.suffix.lower() in ('.png','.jpg'):
            with Image.open(p) as im:
                im.load()
                dims=list(im.size)
            outputs.append({'path':p.name,'bytes':p.stat().st_size,'sha256':digest(p),'size':dims})
    manifest.update(projection_status='rendered',outputs=outputs)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return out


def main():
    parser=argparse.ArgumentParser(description='CPU spherical ERP inspection; fresh results beside input')
    parser.add_argument('--image',required=True,type=Path)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--fov',type=float,default=90,help='Horizontal and vertical FoV in degrees (square views)')
    parser.add_argument('--layout',choices=('eight','six'),default='eight',help='eight: ordered 45-degree yaw ring plus poles (default); six: legacy cardinal views plus poles')
    args=parser.parse_args()
    try:
        out=check_panorama(args.image,args.source,args.size,args.fov,args.layout)
    except (OSError,ValueError,RuntimeError) as exc:
        parser.exit(1,f'Inspection failed: {exc}\n')
    print(json.dumps({'output_dir':str(out),'projection_status':'rendered','review_status':'pending_manual_review'}))

if __name__=='__main__':
    main()
