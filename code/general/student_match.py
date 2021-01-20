import sys
import random


# Python3

Skills={}
Clusters={}

while (True):
    name=input("What is your name? ")
    if name=="###":
        break
    skill=int(input("What is your Python skill level? (1-5) "))
    Skills[name]=skill
    if skill not in Clusters:
        Clusters[skill]=[name]
    elif name not in Clusters[skill]:
        Clusters[skill].append(name)

ClustersL=[Clusters[c] for c in sorted(Clusters)]
Assigned=[]

#print(ClustersL)

while (len(ClustersL)>=2):
    cl1=ClustersL[0]
    cl2=ClustersL[1]
    if len(cl1)<=len(cl2):
        for a in cl1:
            b=cl2.pop(0)
            Assigned.append((a,b))
        del ClustersL[0]
    elif len(cl1)>=len(cl1):
        for a in cl2:
            b=cl1.pop(0)
            Assigned.append((a,b))
        del ClustersL[1]
    
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

f=open("student-pairs.txt","w")
for (a,b) in Assigned:
    print("%s is linked to %s"%(a,b))
    f.write("%s is linked to %s\n"%(a,b))
f.close()
print("See student-pairs.txt")

exit(0)
        


