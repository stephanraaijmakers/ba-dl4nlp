import sys
import itertools
import re

# Stephan Raaijmakers, 2024

from itertools import product, combinations, chain
from more_itertools import pairwise

def all_bracketings(seq):
    if len(seq) <= 1:
        yield from seq
    else:
        for n_children in range(2, len(seq)+1):
            for breakpoints in combinations(range(1, len(seq)), n_children-1):
                children = [seq[i:j] for i,j in pairwise(chain((0,), breakpoints, (len(seq)+1,)))]
                yield from product(*(all_bracketings(child) for child in children))

                
def main():
    x1=int(sys.argv[1])
    x2=int(sys.argv[2])
    x3=int(sys.argv[3])
    x4=int(sys.argv[4])
    operators=['+','-','/','*']
    OP=[]
    for o1 in operators:
        for o2 in operators:
            for o3 in operators:
                OP.append([o1,o2,o3])
    perm=list(itertools.permutations([x1,x2,x3,x4]))
           
    for p in perm:
        for op in OP:
            e=[str(p[0])+op[0],str(p[1])+op[1],str(p[2])+op[2],str(p[3])]
            br=list(all_bracketings(e))
            for b in br:
                b=str(b)
                orig=b
                while True:                    
                    b=re.sub(",","",b)
                    b=re.sub("'","",b)
                    b=re.sub("\+\)",")+",b)
                    b=re.sub("\-\)",")-",b)
                    b=re.sub("/\)",")/",b)
                    b=re.sub("\*\)",")*",b)
                    if b!=orig:
                        orig=b
                    else:
                        break
                try:
                    if eval(b)==24:
                        print("SOLUTION:",b)
                except ZeroDivisionError:
                    True
                  
if __name__=="__main__":
    main()
    
    
        
        
    
