import argparse,concurrent.futures,hashlib,json,os,pathlib,secrets,signal,statistics,subprocess,threading,time,urllib.request
BASE=pathlib.Path('/home/xs/codex-work/ornith-opt-20261006')

def http(base,path,key='',body=None,timeout=20):
 headers={'Content-Type':'application/json'}
 if key:headers['Authorization']='Bearer '+key
 req=urllib.request.Request(base+path,headers=headers,data=None if body is None else json.dumps(body).encode())
 return json.load(urllib.request.urlopen(req,timeout=timeout))

def live():
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=(p/'cmdline').read_bytes().decode().strip('\0').split('\0')
   if args and 'llama-server' in args[0] and '--port' in args and args[args.index('--port')+1]=='8096':
    env=dict(v.split('=',1) for v in (p/'environ').read_bytes().decode().strip('\0').split('\0') if '=' in v)
    return int(p.name),args,env,os.readlink(p/'cwd')
  except (OSError,UnicodeDecodeError):pass
 raise RuntimeError('Live llama-server on port 8096 was not found')

def alive(pid):
 try:return pathlib.Path('/proc/'+str(pid)+'/stat').read_text().split(') ',1)[1][0]!='Z'
 except OSError:return False

def stop(pid):
 if not alive(pid):return
 os.kill(pid,signal.SIGTERM)
 for _ in range(100):
  if not alive(pid):return
  time.sleep(.1)
 os.kill(pid,signal.SIGKILL)
 for _ in range(100):
  if not alive(pid):return
  time.sleep(.1)
 raise RuntimeError('Process did not stop')

def stop_group(proc):
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=15)
 except subprocess.TimeoutExpired:pass
 try:os.killpg(proc.pid,signal.SIGKILL)
 except ProcessLookupError:pass
 time.sleep(.5)

def option(args,name,value):
 args=list(args)
 if name in args:
  i=args.index(name);args[i+1]=str(value)
 else:args.extend([name,str(value)])
 return args

def gpu():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,clocks.current.sm,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
 return [[float(v.strip()) for v in line.split(',')] for line in r.stdout.splitlines() if line.strip()]

def ready(base,proc,timeout=150):
 start=time.monotonic()
 while time.monotonic()-start<timeout:
  if proc.poll() is not None:raise RuntimeError('server exited: '+str(proc.returncode))
  try:
   if http(base,'/health',timeout=2).get('status')=='ok':return time.monotonic()-start
  except Exception:pass
  time.sleep(.5)
 raise TimeoutError('server readiness timeout')

def request(base,key,prompt,output,index,barrier=None):
 body={'prompt':prompt,'n_predict':output,'temperature':0,'seed':100+index,'ignore_eos':True,'cache_prompt':False,'stream':True}
 headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
 if barrier:barrier.wait()
 start=time.perf_counter();first=None;timings=None;text=[]
 req=urllib.request.Request(base+'/completion',headers=headers,data=json.dumps(body).encode())
 with urllib.request.urlopen(req,timeout=900) as r:
  for line in r:
   if not line.startswith(b'data: '):continue
   try:d=json.loads(line[6:])
   except ValueError:continue
   if d.get('content'):
    if first is None:first=time.perf_counter()
    text.append(d['content'])
   if d.get('timings'):timings=d['timings']
   if d.get('error'):raise RuntimeError(str(d['error']))
 value=''.join(text)
 return {'ttft_s':None if first is None else first-start,'elapsed_s':time.perf_counter()-start,'timings':timings,'output_sha256':hashlib.sha256(value.encode()).hexdigest(),'output_preview':value[:120]}

def cohort(base,key,concurrency,input_tokens,output_tokens,label):
 def prompt(i):
  text=('Synthetic benchmark '+label+' request '+str(i)+'. Continue counting integers with commas.\n'+('1,2,3,4,5,6,7,8,9,10.\n'*max(30,input_tokens//10)))
  tokens=http(base,'/tokenize',key,{'content':text,'add_special':True},timeout=60)['tokens']
  if len(tokens)<input_tokens:raise RuntimeError('Token generator too short')
  return tokens[:input_tokens]
 prompts=[prompt(i) for i in range(concurrency)]
 barrier=threading.Barrier(concurrency+1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
  jobs=[pool.submit(request,base,key,prompts[i],output_tokens,i,barrier) for i in range(concurrency)]
  start=time.perf_counter();barrier.wait();rows=[job.result() for job in jobs];wall=time.perf_counter()-start
 total=sum((row.get('timings') or {}).get('predicted_n',0) for row in rows)
 if total!=concurrency*output_tokens:raise RuntimeError('Output-token count mismatch: '+str(total))
 return {'kind':'cohort','concurrency':concurrency,'requested_input_tokens':input_tokens,'output_tokens_per_request':output_tokens,'wall_s':wall,'output_tokens':total,'aggregate_e2e_output_tps':total/wall,'median_ttft_s':statistics.median(row['ttft_s'] for row in rows if row['ttft_s'] is not None),'requests':rows}

def emit(path,data):
 with path.open('a') as f:f.write(json.dumps(data)+'\n')
 print(json.dumps({k:v for k,v in data.items() if k not in ['requests','props']}),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('plan');opts=parser.parse_args()
 plan=json.loads(pathlib.Path(opts.plan).read_text(encoding='utf-8-sig'));run=plan['run'];out=BASE/'results'/(run+'.jsonl')
 pid,original,env,cwd=live();original_key=original[original.index('--api-key')+1] if '--api-key' in original else ''
 slots=http('http://127.0.0.1:8096','/slots',original_key)
 if any(s.get('is_processing') for s in slots):raise RuntimeError('Live service busy; test did not start')
 summary={'kind':'start','run':run,'original_pid':pid,'original_binary':original[0],'original_contexts':sorted(set(s.get('n_ctx') for s in slots)),'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
 emit(out,summary)
 interrupted=False;current=None
 def cancel(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
 signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
 try:
  stop(pid);interrupted=True
  for case in plan['cases']:
   current=None;name=case['name'];key=secrets.token_urlsafe(24);base='http://127.0.0.1:8097'
   args=original.copy();args[0]=case.get('binary',original[0])
   for flag,value in {'--host':'127.0.0.1','--port':8097,'--api-key':key,**case.get('options',{})}.items():args=option(args,flag,value)
   args.extend(case.get('flags',[]))
   command=case.get('prefix',[])+args
   logpath=BASE/'logs'/(run+'_'+name+'.log');record={'kind':'case_start','case':name,'binary':args[0],'options':case.get('options',{}),'flags':case.get('flags',[])};emit(out,record)
   try:
    with logpath.open('w') as log:
     current=subprocess.Popen(command,cwd=cwd,env={**env,**case.get('env',{})},stdout=log,stderr=log,start_new_session=True)
     load=ready(base,current);slots=http(base,'/slots',key);contexts=sorted(set(s.get('n_ctx') for s in slots))
     if contexts!=[240128]:raise RuntimeError('240K slot limit was not preserved: '+str(contexts))
     emit(out,{'kind':'case_ready','case':name,'startup_s':load,'contexts':contexts,'slots':len(slots),'idle_gpu':gpu()})
     samples=[];sample_stop=threading.Event()
     def sampler():
      while not sample_stop.is_set():
       try:samples.append({'t':time.time(),'gpus':gpu()})
       except Exception:pass
       sample_stop.wait(1)
     monitor=threading.Thread(target=sampler,daemon=True);monitor.start()
     try:
      warm=cohort(base,key,min(case.get('warm_concurrency',16),len(slots)),128,32,'warm')
      emit(out,{'kind':'warmup','case':name,'wall_s':warm['wall_s']})
      for spec in case.get('tests',plan.get('tests',[])):
       for rep in range(spec.get('repeats',1)):
        row=cohort(base,key,spec['concurrency'],spec['input_tokens'],spec['output_tokens'],str(spec['input_tokens'])+'-'+str(rep))
        row.update({'case':name,'repeat':rep});emit(out,row)
     finally:
      sample_stop.set();monitor.join(timeout=6)
      (BASE/'results'/(run+'_'+name+'_gpu.json')).write_text(json.dumps(samples))
     emit(out,{'kind':'case_complete','case':name,'health':http(base,'/health')})
   except Exception as e:
    emit(out,{'kind':'case_failed','case':name,'error_type':type(e).__name__,'error':str(e),'log_tail':logpath.read_text(errors='replace')[-3500:] if logpath.exists() else ''})
   finally:
    if current is not None:stop_group(current)
    current=None
 except BaseException as e:
  emit(out,{'kind':'run_interrupted','error_type':type(e).__name__,'error':str(e)})
 finally:
  if current is not None:stop_group(current)
  if interrupted:
   with open('/tmp/crack8096.log','a') as log:
    restored=subprocess.Popen(original,cwd=cwd,env=env,stdout=log,stderr=log,start_new_session=True)
    try:
     load=ready('http://127.0.0.1:8096',restored,180)
     emit(out,{'kind':'restored','pid':restored.pid,'startup_s':load,'health':http('http://127.0.0.1:8096','/health'),'contexts':sorted(set(s.get('n_ctx') for s in http('http://127.0.0.1:8096','/slots',original_key)))})
    except Exception as e:emit(out,{'kind':'restore_failed','error':str(e),'pid':restored.pid});raise
 emit(out,{'kind':'run_complete','run':run})
if __name__=='__main__':main()
