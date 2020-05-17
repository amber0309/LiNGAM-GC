import numpy as np
from lingamgc import LiNGAM_GC
from gendata import gen_GCM

def demo():
	X, B, k= gen_GCM(4, 3)

	mdl = LiNGAM_GC()
	mdl.fit(X)
	mdl.get_results()

	print('\nTrue causal order')
	print(k)
	print('The true graph structure is:')
	print(B)

if __name__ == '__main__':
	demo()