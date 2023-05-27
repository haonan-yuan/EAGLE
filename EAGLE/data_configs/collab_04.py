import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data')
processed_datafile = f"{dataroot}/collab_04"

dataset='collab_04'
testlength=5
vallength=1