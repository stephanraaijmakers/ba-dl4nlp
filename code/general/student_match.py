import sys
import re
import random
import collections

# Python3

Skills={}
Clusters={}

while (True):
    name=input("What is your name? ")
    if name=="stop":
        break
    skill=int(input("What is your Python skill level? (1-5) "))
    Skills[name]=skill
    if skill not in Clusters:
        Clusters[skill]=[name]
    elif name not in Clusters[skill]:
        Clusters[skill].append(name)

ClustersL=[Clusters[c] for c in sorted(Clusters)]
Assigned=[]

while (len(ClustersL)>=2):
    tmp=ClustersL
    Remove=[]
    cl1=tmp[0]
    cl2=tmp[1]
    if len(cl1)<=len(cl2):
        for a in cl1:
            b=cl2.pop(0)
            Assigned.append((a,b))
        Remove.append(0)
    elif len(cl1)>=len(cl1):
        for a in cl2:
            b=cl1.pop(0)
            Assigned.append((a,b))
        Remove.append(1)
    for x in Remove:
        del tmp[x]
    ClustersL=tmp

tried={}
if len(ClustersL[0]) % 2 !=0:
       ClustersL[0].append('none')
for x in ClustersL[0]:
    for y in ClustersL[0]:
        if x==y:
            continue
        if x in tried or y in tried:
            continue
        Assigned.append((x,y))
        tried[x]=1
        tried[y]=1


print(Assigned)
exit(0)
        


