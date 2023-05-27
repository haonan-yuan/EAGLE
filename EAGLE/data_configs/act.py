import os,sys
CUR_DIR= os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR,'../../data')
processed_datafile = f"{dataroot}/act"

dataset = 'act'
testlength = 8
vallength = 2
length = 30