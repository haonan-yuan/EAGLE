import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data')
processed_datafile = f"{dataroot}/yelp"

dataset='yelp'
testlength=8
vallength=1
length=24
shift=3972
num_nodes=13095