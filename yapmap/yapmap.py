import os,tempfile,multiprocessing as mp
import numpy as np,pandas as pd
from tqdm import tqdm

# default num proc is?
DEFAULT_NUM_PROC = mp.cpu_count()

def pmap_iter(func, objs, args=[], kwargs={}, num_proc=DEFAULT_NUM_PROC, use_threads=False, progress=True, desc=None, **y):
	"""
	Yields results of func(obj) for each obj in objs
	Uses multiprocessing.Pool(num_proc) for parallelism.
	If use_threads, use ThreadPool instead of Pool.
	Results in any order.
	"""
	
	# check num proc
	num_cpu = mp.cpu_count()
	if num_proc>num_cpu: num_proc=num_cpu

	# if parallel
	if not desc: desc=f'Mapping {func.__name__}()'
	if desc: desc=f'{desc} [x{num_proc}]'
	if num_proc>1 and len(objs)>1:

		# real objects
		objects = [(func,obj,args,kwargs) for obj in objs]

		# create pool
		pool=mp.Pool(num_proc) if not use_threads else mp.pool.ThreadPool(num_proc)

		# yield iter
		iterr = pool.imap(_pmap_do, objects)
		
		for res in tqdm(iterr,total=len(objects),desc=desc) if progress else iterr:
			yield res

		# Close the pool?
		pool.close()
		pool.join()
	else:
		# yield
		for obj in (tqdm(objs,desc=desc) if progress else objs):
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




"""
Pandas functions
"""

def pmap_df(df, func, num_proc=DEFAULT_NUM_PROC):
	df_split = np.array_split(df, num_proc)
	df = pd.concat(pmap(func, df_split, num_proc=num_proc))
	return df


def pmap_groups(func,df_grouped,use_cache=True,num_proc=DEFAULT_NUM_PROC,**attrs):


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
		tmpdir=tempfile.mkdtemp()
		for i,(group_name,group_df) in enumerate(tqdm(list(df_grouped),desc='Preparing input')):
			tmp_path = os.path.join(tmpdir, str(i)+'.pkl')
			# print([i,group_name,tmp_path,group_df])
			group_df.to_pickle(tmp_path)
			objs+=[(func,tmp_path,group_key,group_name)]

	# desc?
	if not attrs.get('desc'): attrs['desc']=f'Mapping {func.__name__}'


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