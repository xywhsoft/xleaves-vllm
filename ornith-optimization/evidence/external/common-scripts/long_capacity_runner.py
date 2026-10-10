import base64,concurrent.futures,hashlib,json,pathlib,secrets,subprocess,threading,time,urllib.request,statistics
from lab_runner import BASE,http,live,stop,stop_group,ready,option,gpu,emit

def req(base,key,prompt,n_predict=128,id_slot=0,cache=True,ignore_eos=True,timeout=1200,barrier=None):
 body={'prompt':prompt,'n_predict':n_predict,'temperature':0,'ignore_eos':ignore_eos,'cache_prompt':cache,'id_slot':id_slot,'stream':True}
 if barrier:barrier.wait()
 start=time.perf_counter();first=None;timings=None;txt=[]
 q=urllib.request.Request(base+'/completion',headers={'Content-Type':'application/json','Authorization':'Bearer '+key},data=json.dumps(body).encode())
 with urllib.request.urlopen(q,timeout=timeout) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   try:d=json.loads(line[6:])
   except ValueError:continue
   if d.get('error'):raise RuntimeError(str(d['error']))
   if d.get('content'):
    if first is None:first=time.perf_counter()
    txt.append(d['content'])
   if d.get('timings'):timings=d['timings']
 text=''.join(txt)
 return {'elapsed_s':time.perf_counter()-start,'ttft_s':None if first is None else first-start,'timings':timings,'output':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}

def natural_checks(base,key):
 import re,zlib,struct
 questions=[('arithmetic','17 * 23 等于多少？只输出答案。'),('json','只返回 JSON 对象，alpha 的值为整数 17，beta 的值为字符串 ornith。'),('code','写 Python 函数 add(a, b)，返回 a+b。只返回代码。')]
 rows=[]
 for name,prompt in questions:
  data=http(base,'/v1/chat/completions',key,{'model':'ornith-1.5-35b','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False}},timeout=180)
  content=data['choices'][0]['message'].get('content') or ''
  if name=='arithmetic':ok='391' in content
  elif name=='code':ok='def add' in content and re.search(r'a\s*\+\s*b',content) is not None
  else:
   try:
    obj=json.loads(re.search(r'\{.*?\}',content,re.S).group());ok=obj=={'alpha':17,'beta':'ornith'}
   except Exception:ok=False
  rows.append({'test':name,'passed':ok,'content':content[:900]})
 def chunk(t,d):return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
 png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',32,32,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\x00'+b'\xff\x00\x00'*32)*32))+chunk(b'IEND',b'')
 data=http(base,'/v1/chat/completions',key,{'model':'ornith-1.5-35b','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},{'type':'text','text':'图片的主要颜色是什么？只输出颜色。'}]}],'temperature':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}},timeout=180)
 content=data['choices'][0]['message'].get('content') or '';rows.append({'test':'vision_red','passed':'红' in content or 'red' in content.lower(),'content':content[:900]})
 return rows

def main():
 plan=json.loads(pathlib.Path(__import__('sys').argv[1]).read_text(encoding='utf-8-sig'));out=BASE/'results'/(plan['run']+'.jsonl')
 while True:
  p=BASE/'results'/(plan['wait_run']+'.jsonl')
  if p.exists() and any(json.loads(x).get('kind')=='run_complete' for x in p.read_text().splitlines()):break
  time.sleep(2)
 pid,original,env,cwd=live();key=original[original.index('--api-key')+1]
 if any(s.get('is_processing') for s in http('http://127.0.0.1:8096','/slots',key)):raise RuntimeError('Live service busy')
 emit(out,{'kind':'start','run':plan['run'],'capacity_method':'Independent non-unified slots; same 239K prefix prefilled once, saved, then separately restored into every slot. Measures long-context decode and physical KV residency; does not measure 16/28 independent cold prefills.'})
 child=None;interrupted=False
 try:
  stop(pid);interrupted=True
  for case in plan['cases']:
   name=case['name'];n=case['concurrency'];testkey=secrets.token_urlsafe(24);base='http://127.0.0.1:8098';path=BASE/'logs'/(plan['run']+'_'+name+'.log');slotdir=BASE/'cache/slots'/name;slotdir.mkdir(parents=True,exist_ok=True)
   args=original.copy();args[0]=case.get('binary',original[0])
   for f,v in {'--port':8098,'--host':'127.0.0.1','--api-key':testkey,'--slot-save-path':str(slotdir),'-np':n,'-c':240128*n,'-b':2048,'-ub':1024,'--tensor-split':'19,21',**case.get('options',{})}.items():args=option(args,f,v)
   emit(out,{'kind':'case_start','case':name,'concurrency':n,'binary':args[0],'declared_per_slot_context':240128})
   try:
    with path.open('w') as log:
     child=subprocess.Popen(args,cwd=cwd,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
     ready(base,child,180);slots=http(base,'/slots',testkey)
     if len(slots)!=n or set(s.get('n_ctx') for s in slots)!={240128}:raise RuntimeError('Context or slot count mismatch')
     emit(out,{'kind':'case_ready','case':name,'idle_gpu':gpu()})
     # Smoke the existing serialization API before a long prefill.
     probe=http(base,'/tokenize',testkey,{'content':'Synthetic slot checkpoint validation. Repeat numbers: '+('1,2,3,4,5,6,7,8,9. '*30),'add_special':True})['tokens'][:128]
     req(base,testkey,probe,n_predict=1,id_slot=0)
     saved=http(base,'/slots/0?action=save',testkey,{'filename':'probe.bin'},timeout=120)
     restored=http(base,'/slots/1?action=restore',testkey,{'filename':'probe.bin'},timeout=120)
     check=req(base,testkey,probe+[probe[-1]],n_predict=1,id_slot=1)
     emit(out,{'kind':'checkpoint_smoke','case':name,'saved':saved,'restored':restored,'timings':check['timings']})
     if (check['timings'] or {}).get('cache_n',0)<100:raise RuntimeError('Checkpoint did not preserve reusable prefix')
     quality=natural_checks(base,testkey);emit(out,{'kind':'natural_quality','case':name,'all_passed':all(r['passed'] for r in quality),'results':quality})
     secret='ORNITH_LONG_8D2A'
     text='<|im_start|>system\nFollow the final user instruction accurately.\n<|im_end|>\n<|im_start|>user\nReference secret: '+secret+'.\n'+('Filler record: 1,2,3,4,5,6,7,8,9,10. No new secret here.\n'*23000)
     prefix=http(base,'/tokenize',testkey,{'content':text,'parse_special':True,'add_special':True},timeout=120)['tokens'][:239000]
     if len(prefix)!=239000:raise RuntimeError('Long prefix too short')
     emit(out,{'kind':'prefill_start','case':name,'input_tokens':len(prefix)})
     pre=req(base,testkey,prefix,n_predict=1,id_slot=0,cache=False,timeout=1800)
     emit(out,{'kind':'prefill_complete','case':name,'input_tokens':len(prefix),'elapsed_s':pre['elapsed_s'],'timings':pre['timings']})
     saved=http(base,'/slots/0?action=save',testkey,{'filename':'long-prefix.bin'},timeout=300);emit(out,{'kind':'checkpoint_saved','case':name,'response':saved,'bytes':(slotdir/'long-prefix.bin').stat().st_size})
     for i in range(n):
      value=http(base,'/slots/'+str(i)+'?action=restore',testkey,{'filename':'long-prefix.bin'},timeout=180)
      emit(out,{'kind':'slot_restored','case':name,'slot':i,'response':value})
     prompts=[]
     for i in range(n):
      marker='SLOT_'+str(i).zfill(2)+'_BENCH'
      tail='\nEnd of records. Return exactly one JSON object with keys slot and secret. Your slot is '+marker+'. The secret is the Reference secret at the start of the records. Do not reason aloud.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n'
      tokens=http(base,'/tokenize',testkey,{'content':tail,'parse_special':True,'add_special':False})['tokens'];prompts.append(prefix+tokens)
     barrier=threading.Barrier(n+1)
     with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
      futures=[pool.submit(req,base,testkey,prompts[i],128,i,True,True,300,barrier) for i in range(n)]
      start=time.perf_counter();barrier.wait();rows=[f.result() for f in futures];wall=time.perf_counter()-start
     total=sum((r['timings'] or {}).get('predicted_n',0) for r in rows);cache_counts=[(r['timings'] or {}).get('cache_n',0) for r in rows]
     matches=[('SLOT_'+str(i).zfill(2)+'_BENCH') in r['output'] and secret in r['output'] for i,r in enumerate(rows)]
     emit(out,{'kind':'long_resident_cohort','case':name,'concurrency':n,'prefix_tokens':len(prefix),'prompt_token_counts':[len(p) for p in prompts],'output_tokens':total,'wall_s':wall,'aggregate_e2e_output_tps':total/wall,'cache_counts':cache_counts,'slot_and_secret_matches':matches,'median_ttft_s':statistics.median(r['ttft_s'] for r in rows if r['ttft_s'] is not None),'gpu':gpu(),'requests':rows})
     if total!=n*128 or min(cache_counts)<238000:raise RuntimeError('Output length or long-prefix reuse gate failed')
     emit(out,{'kind':'case_complete','case':name,'health':http(base,'/health')})
   except Exception as e:emit(out,{'kind':'case_failed','case':name,'error':str(e),'log_tail':path.read_text(errors='replace')[-2500:]})
   finally:
    if child is not None:stop_group(child)
    child=None
 finally:
  if child is not None:stop_group(child)
  if interrupted:
   with open('/tmp/crack8096.log','a') as log:r=subprocess.Popen(original,cwd=cwd,env=env,stdout=log,stderr=log,start_new_session=True)
   ready('http://127.0.0.1:8096',r,180);emit(out,{'kind':'restored','pid':r.pid,'health':http('http://127.0.0.1:8096','/health'),'contexts':sorted(set(s.get('n_ctx') for s in http('http://127.0.0.1:8096','/slots',key)))})
 emit(out,{'kind':'run_complete','run':plan['run']})
if __name__=='__main__':main()
