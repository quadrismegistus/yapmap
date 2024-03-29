import os,tempfile,multiprocess as mp,sys,random
import numpy as np,pandas as pd

CONTEXT='fork'
# default num proc is?
mp_cpu_count=mp.cpu_count()
if mp_cpu_count==1: DEFAULT_NUM_PROC=1
elif mp_cpu_count==2: DEFAULT_NUM_PROC=2
elif mp_cpu_count==3: DEFAULT_NUM_PROC=2
else: DEFAULT_NUM_PROC = mp_cpu_count - 2


def in_jupyter(): return sys.argv[-1].endswith('json')

def get_tqdm(iterable,*args,**kwargs):
    l=iterable
    if type(l)==list and len(l)==1: return l
    if in_jupyter():
        from tqdm import tqdm as tqdmx
    else:
        from tqdm import tqdm as tqdmx
    return tqdmx(l,*args,**kwargs)



tqdm = get_tqdm


def pmap_iter(
        func, 
        objs, 
        args=[], 
        kwargs={}, 
        lim=None,
        num_proc=DEFAULT_NUM_PROC, 
        use_threads=False, 
        progress=True, 
        progress_pos=0,
        desc=None,
        context=CONTEXT, 
        **y):
    """
    Yields results of func(obj) for each obj in objs
    Uses multiprocessing.Pool(num_proc) for parallelism.
    If use_threads, use ThreadPool instead of Pool.
    Results in any order.
    """

    # lim?
    if lim: objs = objs[:lim]

    # check num proc
    num_cpu = mp.cpu_count()
    if num_proc>num_cpu: num_proc=num_cpu
    if num_proc>len(objs): num_proc=len(objs)

    # if parallel
    if not desc: desc=f'Mapping {func.__name__}()'
    if desc and num_cpu>1: desc=f'{desc} [x{num_proc}]'
    if num_proc>1 and len(objs)>1:

        # real objects
        objects = [(func,obj,args,kwargs) for obj in objs]

        # create pool
        #pool=mp.Pool(num_proc) if not use_threads else mp.pool.ThreadPool(num_proc)
        with mp.get_context(context).Pool(num_proc) as pool:
            # yield iter
            iterr = pool.imap(_pmap_do, objects)

            for res in get_tqdm(iterr,total=len(objects),desc=desc,position=progress_pos) if progress else iterr:
                yield res
    else:
        # yield
        for obj in (tqdm(objs,desc=desc,position=progress_pos) if progress else objs):
            yield func(obj,*args,**kwargs)

def _pmap_do(inp):
    func,obj,args,kwargs = inp
    return func(obj,*args,**kwargs)

def pmap(*x,**y):
    """
    Non iterator version of pmap_iter
    """
    # return as list
    return list(pmap_iter(*x,**y))

def pmap_run(*x,**y):
    for obj in pmap_iter(*x,**y): pass


"""
Pandas functions
"""

def pmap_df(df, func, num_proc=DEFAULT_NUM_PROC):
    df_split = np.array_split(df, num_proc)
    df = pd.concat(pmap(func, df_split, num_proc=num_proc))
    return df


def pmap_groups(func,df_grouped,use_cache=True,num_proc=DEFAULT_NUM_PROC,shuffle=True,**attrs):


    # get index/groupby col name(s)
    group_key=df_grouped.grouper.names
    # if not using cache
    # if not use_cache or attrs.get('num_proc',1)<2:
    if not use_cache or len(df_grouped)<2 or num_proc<2:
        objs=[
            (func,group_df,group_key,group_name)
            for group_name,group_df in df_grouped
        ]
    else:
        objs=[]
        groups=list(df_grouped)
        for i,(group_name,group_df) in enumerate(tqdm(groups,desc='Preparing input')):
            # print([i,group_name,tmp_path,group_df])
            if use_cache:
                tmpdir=tempfile.mkdtemp()
                tmp_path = os.path.join(tmpdir, str(i)+'.pkl')
                group_df.to_pickle(tmp_path)
            objs+=[(func,tmp_path if use_cache else group_df,group_key,group_name)]

    # desc?
    if not attrs.get('desc'): attrs['desc']=f'Mapping {func.__name__}'

    if shuffle: random.shuffle(objs)

    return pd.concat(
        pmap(
            _do_pmap_group,
            objs,
            num_proc=num_proc,
            **attrs
        )
    ).set_index(group_key)




def _do_pmap_group(obj,*x,**y):
    # unpack
    func,group_df,group_key,group_name = obj
    # load from cache?
    if type(group_df)==str:
        group_df=pd.read_pickle(group_df)
    # run func
    outdf=func(group_df,*x,**y)
    # annotate with groupnames on way out
    if type(group_name) not in {list,tuple}:group_name=[group_name]
    for x,y in zip(group_key,group_name):
        outdf[x]=y
    # return
    return outdf



def pmap_apply_cols(func, df, lim=None, **y):
    cols=list(df.columns)[:lim]
    new_seriess = pmap(
        func,
        [df[col] for col in cols],
        **y
    )
    odf=pd.DataFrame(dict(zip(cols,new_seriess)))
    return odf
