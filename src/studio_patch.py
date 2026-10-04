"""Apply the current head-trainer interaction to the dataset studio."""

import dataset_studio as studio


immediate_handler = r'''function boxEventsImmediate(){document.querySelector('nav').style.display='none';document.querySelector('main').style.maxWidth='none';document.querySelector('main').style.padding='0';document.querySelector('#app > h1').style.display='none';document.querySelector('#app > .lede').style.display='none';document.querySelector('#app > .statgrid').style.display='none';let layout=document.querySelector('.split'),panel=document.querySelector('.split .panel:first-child'),side=document.querySelector('.split .panel:last-child'),stage=document.querySelector('#stage'),image=document.querySelector('#image'),boxes=document.querySelector('#boxes'),save=document.querySelector('#save'),first=null,current=null,saved=false;layout.style.display='block';panel.style.border='0';panel.style.padding='0';panel.style.boxShadow='none';side.style.display='none';stage.style.width='100vw';image.style.width='100vw';image.style.maxWidth='100vw';image.style.maxHeight='calc(100vh - 70px);function point(e){let bounds=image.getBoundingClientRect(),scale=bounds.width/image.offsetWidth;return{x:Math.max(0,Math.min(image.offsetWidth,(e.clientX-bounds.left)/scale)),y:Math.max(0,Math.min(image.offsetHeight,(e.clientY-bounds.top)/scale))}}function redraw(end){let left=Math.min(first.x,end.x),top=Math.min(first.y,end.y);current.style.left=left+'px';current.style.top=top+'px';current.style.width=Math.abs(first.x-end.x)+'px';current.style.height=Math.abs(first.y-end.y)+'px'}async function cancel(){if(saved){await api('/api/head/undo',{method:'POST'});saved=false}if(current)current.remove();first=null;current=null;save.disabled=true;document.querySelector('#message').textContent='Box undone. Click the first corner of a head.'}image.draggable=false;stage.style.width='100%';image.style.width='100%';image.style.maxHeight='calc(100vh - 190px)';stage.addEventListener('dragstart',e=>e.preventDefault());document.addEventListener('keydown',e=>{if(e.key==='Escape')cancel()});image.addEventListener('mousemove',e=>{if(first)redraw(point(e))});image.addEventListener('click',async e=>{e.preventDefault();let clicked=point(e);if(!first){first=clicked;current=document.createElement('i');current.className='box';boxes.appendChild(current);current.style.left=clicked.x+'px';current.style.top=clicked.y+'px';document.querySelector('#message').textContent='First corner set. Move the mouse to preview, then click the opposite corner.';return}let end=clicked;redraw(end);let box={x1:Math.min(first.x,end.x)/image.offsetWidth,y1:Math.min(first.y,end.y)/image.offsetHeight,x2:Math.max(first.x,end.x)/image.offsetWidth,y2:Math.max(first.y,end.y)/image.offsetHeight};if(box.x2-box.x1<.01||box.y2-box.y1<.01){await cancel();return}first=null;save.disabled=true;await api('/api/head/box',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(box)});saved=true;current.className='box done';document.querySelector('#n').textContent++;document.querySelector('#message').textContent='Box accepted. Press Escape to undo it, or click the first corner of the next head.'});document.querySelector('#next').onclick=heads}'''

studio.PAGE = studio.PAGE.replace(';boxEvents()}', ';boxEventsImmediate()}')
studio.PAGE = studio.PAGE.replace("image.style.maxHeight='calc(100vh - 70px);", "image.style.maxHeight='calc(100vh - 70px)';")
studio.PAGE = studio.PAGE.replace("image.style.maxHeight='calc(100vh - 190px)'", "image.style.maxHeight='calc(100vh - 70px)'")
studio.PAGE = studio.PAGE.replace(
	'<button class="gray" id="next">Skip image</button>',
	'<button class="gray" id="skip">Skip image</button><button class="gray" id="next">Next image</button>',
)
immediate_handler = immediate_handler.replace(
	"document.querySelector('#next').onclick=heads}",
	"document.querySelector('#next').onclick=heads;document.querySelector('#skip').onclick=async()=>{await api('/api/head/skip',{method:'POST'});heads()}}",
)
studio.PAGE = studio.PAGE.replace('async function labels()', immediate_handler + 'async function labels()', 1)
studio.PAGE = studio.PAGE.replace("image.style.maxHeight='calc(100vh - 70px);", "image.style.maxHeight='calc(100vh - 70px)';")
studio.PAGE = studio.PAGE.replace("image.style.maxHeight='calc(100vh - 190px)'", "image.style.maxHeight='calc(100vh - 70px)'")

main = studio.main
