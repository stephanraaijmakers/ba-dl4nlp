import sys

with open(sys.argv[1], "r") as fp:
    lines=fp.readlines()

n=1    
for line in lines:
    print(n)
    f=open("sample-%d.txt"%(n),"w")
    f.write(line)
    f.close()
    n+=1
    
   
