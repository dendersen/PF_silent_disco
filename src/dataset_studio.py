"""Browser dataset studio for head, presence, and color training."""

from __future__ import annotations

import argparse
import base64
import json
import random
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import torch

from silent_disco import COLOR_MODEL_CLASSES, crop_255, detect_people, load_model, load_person_detector, person_head_crop_box, predict_probabilities

LABELS = ("green", "blue", "red", "no-headset", "no-head")

PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Silent Disco Dataset Studio</title><style>
:root{--ink:#17202a;--muted:#687385;--line:#d9dee7;--bg:#f4f5f7;--paper:#fff;--accent:#e4572e;--green:#16875b;--blue:#2864d7;--red:#c83c4a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 Georgia,serif}main{max-width:1180px;margin:auto;padding:25px 20px 60px}nav{display:flex;gap:16px;align-items:center;border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:25px}nav a{color:var(--ink);text-decoration:none;font-weight:bold}nav a:first-child{margin-right:auto;color:var(--accent);font-size:20px}h1{font-size:42px;margin:8px 0}h2{margin-top:0}.lede,.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:15px}.card,.panel{background:var(--paper);border:1px solid var(--line);padding:18px;box-shadow:0 4px 18px #17202a0c}.statgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:8px;margin:20px 0}.stat{background:var(--paper);border-top:3px solid var(--accent);padding:10px}.stat strong{display:block;font-size:27px}.stat span{color:var(--muted);font-size:13px}.buttonbar{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}button,.button{border:0;background:var(--ink);color:white;padding:11px 14px;cursor:pointer;font:600 15px Georgia,serif;text-decoration:none}button:disabled{opacity:.45}.accent{background:var(--accent)}.green{background:var(--green)}.blue{background:var(--blue)}.red{background:var(--red)}.gray{background:#687385}.active{outline:4px solid #f0b429;outline-offset:2px}.split{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:20px}.stage{position:relative;display:inline-block;background:#111;max-width:100%}.stage img{display:block;max-width:100%;max-height:70vh}.boxes{position:absolute;inset:0;pointer-events:none}.box{position:absolute;border:3px solid var(--accent);background:#e4572e33}.done{border-color:var(--green);background:#16875b33}.crop{width:360px;max-width:100%;background:#111}.notice{padding:10px;border-left:4px solid #f0b429;background:#fff5df}.log{background:#17202a;color:#e3eaf0;white-space:pre-wrap;min-height:170px;padding:12px;font:13px monospace}@media(max-width:800px){h1{font-size:32px}.split{grid-template-columns:1fr}}
</style></head><body><main><nav><a href="/">DATASET STUDIO</a><a href="/heads">Head trainer</a><a href="/labels">Color + presence</a></nav><div id="app"></div></main><script>
const app=document.querySelector('#app');async function api(p,o){let r=await fetch(p,o);if(!r.ok)throw Error(await r.text());return r.status===204?null:r.json()}function stats(s){return '<div class="statgrid">'+Object.entries(s).map(([k,v])=>`<div class="stat"><strong>${v}</strong><span>${k.replaceAll('_',' ')}</span></div>`).join('')+'</div>'}async function home(){let s=await api('/api/stats');app.innerHTML=`<h1>Train the view.</h1><p class="lede">Build the head detector, then label what the event really contains. Video frames are prioritized because they represent the live event better than still photos.</p>${stats(s)}<div class="grid"><section class="card"><h2>Head trainer</h2><p>Draw one box around each visible head. YOLO labels are saved for detector training.</p><a class="button accent" href="/heads">Open head trainer</a></section><section class="card"><h2>Color + presence</h2><p>Review one detector candidate at a time. AI predictions are highlighted and scored against your answers.</p><a class="button" href="/labels">Open classifier trainer</a></section></div><section class="card" style="margin-top:15px"><h2>CPU training</h2><p class="muted">Every job uses CPU and stops at 15 epochs. Stop it whenever you need.</p><div class="buttonbar"><button onclick="train('head')">Train head detector</button><button onclick="train('presence')">Train presence</button><button onclick="train('color')">Train color</button><button class="gray" onclick="stop()">Stop</button></div><div id="job"></div><pre id="log" class="log">No training output yet.</pre></section>`;poll()}async function train(k){await api('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k})});poll()}async function stop(){await api('/api/train/stop',{method:'POST'});poll()}async function poll(){let j=await api('/api/train/status'),l=document.querySelector('#log');if(!l)return;l.textContent=j.log||'No training output yet.';document.querySelector('#job').textContent=j.running?'Training '+j.kind+'...':j.status;if(j.running)setTimeout(poll,800)}
async function heads(){let d=await api('/api/head/frame');app.innerHTML=`<h1>Head trainer</h1><p class="lede">${d.note}</p>${stats(d.stats)}<div class="split"><section class="panel"><div class="stage" id="stage"><img id="image" src="data:image/jpeg;base64,${d.image}"><div class="boxes" id="boxes"></div></div><div class="buttonbar"><button class="accent" id="save" disabled>Save box</button><button class="gray" id="next">Skip image</button></div><p id="message" class="muted">Drag a box around one head at a time.</p></section><section class="panel"><h2>${d.name}</h2><p>Saved boxes in this session: <b id="n">0</b></p><p class="notice">Video frames are selected about 80% of the time. Skips are not saved.</p></section></div>`;boxEvents()}
function boxEvents(){let layout=document.querySelector('.split'),side=document.querySelector('.split .panel:last-child'),stage=document.querySelector('#stage'),image=document.querySelector('#image'),boxes=document.querySelector('#boxes'),save=document.querySelector('#save'),first=null,current=null;layout.style.gridTemplateColumns='1fr';side.style.display='none';function point(e){let bounds=image.getBoundingClientRect(),scale=bounds.width/image.offsetWidth;return{x:Math.max(0,Math.min(image.offsetWidth,(e.clientX-bounds.left)/scale)),y:Math.max(0,Math.min(image.offsetHeight,(e.clientY-bounds.top)/scale))}}function redraw(end){let left=Math.min(first.x,end.x),top=Math.min(first.y,end.y);current.style.left=left+'px';current.style.top=top+'px';current.style.width=Math.abs(first.x-end.x)+'px';current.style.height=Math.abs(first.y-end.y)+'px'}function cancel(){if(current)current.remove();first=null;current=null;save.disabled=true;document.querySelector('#message').textContent='Box cancelled. Click the first corner of a head.'}image.draggable=false;stage.style.width='100%';image.style.width='100%';image.style.maxHeight='calc(100vh - 190px)';stage.addEventListener('dragstart',e=>e.preventDefault());document.addEventListener('keydown',e=>{if(e.key==='Escape')cancel()});image.addEventListener('mousemove',e=>{if(first)redraw(point(e))});image.addEventListener('click',e=>{e.preventDefault();let clicked=point(e);if(!first){first=clicked;current=document.createElement('i');current.className='box';boxes.appendChild(current);current.style.left=clicked.x+'px';current.style.top=clicked.y+'px';document.querySelector('#message').textContent='First corner set. Move the mouse to preview, then click the opposite corner.';return}let end=clicked;redraw(end);let v={x1:Math.min(first.x,end.x)/image.offsetWidth,y1:Math.min(first.y,end.y)/image.offsetHeight,x2:Math.max(first.x,end.x)/image.offsetWidth,y2:Math.max(first.y,end.y)/image.offsetHeight};if(v.x2-v.x1<.01||v.y2-v.y1<.01){cancel();return}first=null;save.disabled=false;save.onclick=async()=>{save.disabled=true;await api('/api/head/box',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(v)});current.className='box done';current=null;document.querySelector('#n').textContent++;document.querySelector('#message').textContent='Saved. Click the first corner of the next head.';save.onclick=null}});document.querySelector('#next').onclick=heads}
async function labels(){let d=await api('/api/label/frame');app.innerHTML=`<h1>Color + presence</h1><p class="lede">The highlighted button is the AI result. Choose the user label, or skip uncertain examples.</p>${stats(d.stats)}<div class="split"><section class="panel"><img class="crop" src="data:image/jpeg;base64,${d.crop}"><p>${d.index+1} of ${d.total} candidates · ${d.source}</p><div class="buttonbar">${['green','blue','red','no-headset','no-head','skip'].map(x=>`<button class="${x==='green'?'green':x==='blue'?'blue':x==='red'?'red':'gray'}" data-label="${x}">${x.replace('-',' ')}</button>`).join('')}</div><p>AI: <b>${d.prediction}</b> · cumulative accuracy: <b>${d.accuracy}%</b></p></section><section class="panel"><h2>How this set is built</h2><p>One random video frame is preferred for every four still-image selections. The detector candidate is cropped, then saved to both presence and color datasets when answered.</p><p class="notice">${d.note}</p></section></div>`;document.querySelector(`[data-label="${d.prediction}"]`)?.classList.add('active');document.querySelectorAll('[data-label]').forEach(b=>b.onclick=async()=>{await api('/api/label/answer',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label:b.dataset.label})});labels()})}if(location.pathname==='/heads')heads();else if(location.pathname==='/labels')labels();else home();</script></body></html>'''


def encode(image):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok: raise ValueError('Could not encode image')
    return base64.b64encode(data).decode('ascii')


class App:
    def __init__(self, image_dir: Path, dataset: Path, detector_path: Path, head_detector_path: Path | None, presence_path: Path | None, color_path: Path | None):
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.stills = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
        self.videos = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'})
        if not self.stills and not self.videos: raise ValueError(f'No media found in {image_dir}')
        self.dataset = dataset; self.detector = load_person_detector(detector_path, torch.device('cpu'))
        self.head_detector = load_person_detector(head_detector_path, torch.device('cpu')) if head_detector_path and head_detector_path.exists() else None
        self.presence_model = load_model(presence_path, 2, torch.device('cpu')) if presence_path else None
        self.color_model = load_model(color_path, len(COLOR_MODEL_CLASSES), torch.device('cpu')) if color_path else None
        self.manifest = dataset / 'studio.json'; self.data = json.loads(self.manifest.read_text()) if self.manifest.exists() else {'heads': [], 'labels': []}
        self.lock = threading.Lock(); self.job = None; self.job_kind = ''; self.job_log = []
        self.head_path = None; self.head_image = None; self.label_items = []; self.label_index = 0
        self.next_head(); self.next_label()

    def write(self): self.dataset.mkdir(parents=True, exist_ok=True); self.manifest.write_text(json.dumps(self.data, indent=2))
    def counts(self):
        c = {'head_boxes': len(self.data['heads']), 'green': 0, 'blue': 0, 'red': 0, 'no_headset': 0, 'no_head': 0, 'skipped': 0}
        for x in self.data['labels']: c[x['label'].replace('-', '_')] = c.get(x['label'].replace('-', '_'), 0) + 1
        return c
    def choose(self):
        if self.videos and (not self.stills or random.random() < .8):
            p = random.choice(self.videos); cap = cv2.VideoCapture(str(p)); count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))); n = random.randrange(count); cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, image = cap.read(); cap.release()
            if ok: return p, image, f'{p.name}#frame={n}'
        p = random.choice(self.stills); return p, cv2.imread(str(p)), p.name
    def next_head(self): self.head_path, self.head_image, self.head_source = self.choose()
    def next_label(self):
        p, image, source = self.choose(); detector = self.head_detector or self.detector; people = detect_people(detector, image, confidence=.03, image_size=1280, tile_size=3000); self.label_items = [{'image': image, 'path': p, 'source': source, 'box': b, 'score': s} for b, s in people]; self.label_index = 0
    def crop(self, image, box):
        x1,y1,x2,y2 = person_head_crop_box(box); size=x2-x1; return cv2.resize(crop_255(image,x1+size//2,y1+size//2,size),(255,255),interpolation=cv2.INTER_AREA)
    def stats(self): return self.counts()
    def head_frame(self):
        h,w=self.head_image.shape[:2]; scale=min(1,1600/max(h,w)); image=cv2.resize(self.head_image,(round(w*scale),round(h*scale))) if scale<1 else self.head_image; return {'image':encode(image),'name':self.head_source,'stats':self.stats(),'note':'Videos are selected about 80% of the time.'}
    def save_box(self, v):
        box=[float(v[k]) for k in ('x1','y1','x2','y2')]
        item={'id': max((int(entry.get('id', -1)) for entry in self.data['heads']), default=-1) + 1, 'image':self.head_source, 'box':box}
        self.data['heads'].append(item)
        root=self.dataset/'heads'; (root/'images').mkdir(parents=True,exist_ok=True); (root/'labels').mkdir(parents=True,exist_ok=True)
        stem=self.head_source.replace('#','_').replace('.','_'); cv2.imwrite(str(root/'images'/f'{stem}.jpg'),self.head_image)
        with_file=root/'labels'/f'{stem}.txt'; x1,y1,x2,y2=box; line=f'0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n'; with_file.write_text(with_file.read_text() + line if with_file.exists() else line)
        self.write()
        return item['id']

    def undo_box(self, box_id):
        index=next((index for index, entry in enumerate(self.data['heads']) if int(entry.get('id', -1)) == box_id), None)
        if index is None:
            return
        entry=self.data['heads'].pop(index); source=str(entry['image']); stem=source.replace('#','_').replace('.','_'); root=self.dataset/'heads'; label_path=root/'labels'/f'{stem}.txt'
        remaining=[item for item in self.data['heads'] if item['image'] == source]
        if remaining:
            lines=[]
            for item in remaining:
                x1,y1,x2,y2=item['box']; lines.append(f'0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n')
            label_path.write_text(''.join(lines))
        else:
            label_path.unlink(missing_ok=True); (root/'images'/f'{stem}.jpg').unlink(missing_ok=True)
        self.write()

    def undo_last_box(self):
        if self.data['heads']:
            self.undo_box(int(self.data['heads'][-1].get('id', -1)))

    def skip_head(self):
        current_ids=[int(entry.get('id', -1)) for entry in self.data['heads'] if entry['image'] == self.head_source]
        for box_id in current_ids:
            self.undo_box(box_id)

    PAGE = PAGE.replace(';boxEvents()}', ';boxEventsImmediate()}')
    PAGE = PAGE.replace(
        'async function labels()',
        r'''function boxEventsImmediate(){let stage=document.querySelector('#stage'),image=document.querySelector('#image'),boxes=document.querySelector('#boxes'),save=document.querySelector('#save'),first=null,current=null,saved=false;function point(e){let bounds=image.getBoundingClientRect(),scale=bounds.width/image.offsetWidth;return{x:Math.max(0,Math.min(image.offsetWidth,(e.clientX-bounds.left)/scale)),y:Math.max(0,Math.min(image.offsetHeight,(e.clientY-bounds.top)/scale))}}function redraw(end){let left=Math.min(first.x,end.x),top=Math.min(first.y,end.y);current.style.left=left+'px';current.style.top=top+'px';current.style.width=Math.abs(first.x-end.x)+'px';current.style.height=Math.abs(first.y-end.y)+'px'}async function cancel(){if(saved){await api('/api/head/undo',{method:'POST'});saved=false}if(current)current.remove();first=null;current=null;save.disabled=true;document.querySelector('#message').textContent='Box undone. Click the first corner of a head.'}image.draggable=false;stage.style.width='100%';image.style.width='100%';image.style.maxHeight='calc(100vh - 190px)';stage.addEventListener('dragstart',e=>e.preventDefault());document.addEventListener('keydown',e=>{if(e.key==='Escape')cancel()});image.addEventListener('mousemove',e=>{if(first)redraw(point(e))});image.addEventListener('click',async e=>{e.preventDefault();let clicked=point(e);if(!first){first=clicked;current=document.createElement('i');current.className='box';boxes.appendChild(current);current.style.left=clicked.x+'px';current.style.top=clicked.y+'px';document.querySelector('#message').textContent='First corner set. Move the mouse to preview, then click the opposite corner.';return}let end=clicked;redraw(end);let box={x1:Math.min(first.x,end.x)/image.offsetWidth,y1:Math.min(first.y,end.y)/image.offsetHeight,x2:Math.max(first.x,end.x)/image.offsetWidth,y2:Math.max(first.y,end.y)/image.offsetHeight};if(box.x2-box.x1<.01||box.y2-box.y1<.01){await cancel();return}first=null;save.disabled=true;await api('/api/head/box',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(box)});saved=true;current.className='box done';document.querySelector('#n').textContent++;document.querySelector('#message').textContent='Box accepted. Press Escape to undo it, or click the first corner of the next head.'});document.querySelector('#next').onclick=heads}
    async function labels()''',
    )
    def prediction(self, crop):
        if self.presence_model is None:
            return 'unknown'
        presence = int(torch.argmax(predict_probabilities(self.presence_model, [crop], torch.device('cpu'))[0]).item())
        if presence == 0:
            return 'no-headset'
        if self.color_model is None:
            return 'unknown'
        color = int(torch.argmax(predict_probabilities(self.color_model, [crop], torch.device('cpu'))[0]).item())
        return COLOR_MODEL_CLASSES[color]
    def label_frame(self):
        if not self.label_items: self.next_label()
        if not self.label_items: raise ValueError('No detector candidates found')
        x=self.label_items[self.label_index]; crop=self.crop(x['image'],x['box']); answers=[a for a in self.data['labels'] if a['label']!='skip']; correct=sum(a.get('correct',False) for a in answers); pred=self.prediction(crop); return {'crop':encode(crop),'source':x['source'],'index':self.label_index,'total':len(self.label_items),'prediction':pred,'accuracy':round(100*correct/max(1,len(answers))),'stats':self.stats(),'note':'Using head detector.' if self.head_detector else 'No head.pt found: using person boxes as temporary candidates.'}
    def answer(self, label):
        x=self.label_items[self.label_index]; pred=self.prediction(self.crop(x['image'],x['box'])); self.data['labels'].append({'image':x['source'],'box':x['box'],'label':label,'prediction':pred,'correct':label==pred});
        if label!='skip':
            name=f'{len(self.data["labels"]):06d}.jpg'; crop=self.crop(x['image'],x['box']); presence='negative' if label in {'no-headset','no-head'} else 'headset'; color=label if label in {'green','blue','red'} else 'unknown';
            for task, group in (('presence',presence),('color',color)): (self.dataset/task/group).mkdir(parents=True,exist_ok=True); cv2.imwrite(str(self.dataset/task/group/name),crop)
        self.write(); self.label_index += 1
        if self.label_index>=len(self.label_items): self.next_label()
    def status(self): return {'running':bool(self.job and self.job.poll() is None),'kind':self.job_kind,'log':''.join(self.job_log)[-12000:],'status':'running' if self.job and self.job.poll() is None else ('finished' if self.job else 'idle')}
    def train(self, kind):
        if self.job and self.job.poll() is None:return
        cmd=[sys.executable,'src/train_models.py',str(self.dataset),'--kind','head','--device','cpu','--epochs','15'] if kind=='head' else [sys.executable,'src/train_models.py',str(self.dataset),'--kind',kind,'--device','cpu','--epochs','15']; self.job_kind=kind; self.job_log=[]; self.job=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); threading.Thread(target=self.read,daemon=True).start()
    def read(self):
        if self.job and self.job.stdout:
            for line in self.job.stdout:self.job_log.append(line)
    def stop(self):
        if self.job and self.job.poll() is None:self.job.terminate();self.job_log.append('Stopped by user.\n')


class Handler(BaseHTTPRequestHandler):
    app=None
    def send(self,status,body,typ): self.send_response(status);self.send_header('Content-Type',typ);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        route=urlparse(self.path).path
        try:
            if route=='/api/stats': body=self.app.stats()
            elif route=='/api/head/frame': self.app.next_head(); body=self.app.head_frame()
            elif route=='/api/label/frame': body=self.app.label_frame()
            elif route=='/api/train/status': body=self.app.status()
            else:return self.send(200,PAGE.encode(),'text/html; charset=utf-8')
            self.send(200,json.dumps(body).encode(),'application/json')
        except Exception as e:self.send(500,str(e).encode(),'text/plain')
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0')); payload=json.loads(self.rfile.read(n) or b'{}'); route=urlparse(self.path).path
        try:
            with self.app.lock:
                if route=='/api/head/box':self.app.save_box(payload)
                elif route=='/api/head/undo':self.app.undo_last_box()
                elif route=='/api/head/skip':self.app.skip_head()
                elif route=='/api/label/answer':self.app.answer(payload['label'])
                elif route=='/api/train':self.app.train(payload['kind'])
                elif route=='/api/train/stop':self.app.stop()
                else:return self.send(404,b'Not found','text/plain')
            self.send(204,b'','')
        except Exception as e:self.send(500,str(e).encode(),'text/plain')
    def log_message(self,*args):pass


def main():
    p=argparse.ArgumentParser();p.add_argument('image_dir',type=Path);p.add_argument('--dataset',type=Path,default=Path('dataset'));p.add_argument('--detector-model',type=Path,default=Path('models/yolo11n.pt'));p.add_argument('--head-detector-model',type=Path,default=Path('models/head-detector/weights/best.pt'));p.add_argument('--presence-model',type=Path,default=Path('models/presence.pt'));p.add_argument('--color-model',type=Path,default=Path('models/color.pt'));p.add_argument('--port',type=int,default=8765);p.add_argument('--open',action='store_true');a=p.parse_args();Handler.app=App(a.image_dir,a.dataset,a.detector_model,a.head_detector_model,a.presence_model,a.color_model);server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);url=f'http://127.0.0.1:{a.port}/';print(f'Open {url} to use the dataset studio.');
    if a.open:webbrowser.open(url)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
