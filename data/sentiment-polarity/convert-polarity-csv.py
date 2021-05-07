import sys
import re


if __name__=="__main__":
    f=open(sys.argv[1],"r")
    print("review,label")
    for line in f:
        line=line.rstrip()
        line=re.sub("\"","",line)
        m=re.match("^(.+)[\s\t]+\.[\s\t]+([^\s]+).*$",line)
        if m:
            label=m.group(2)
            if label=="0":
                label="neg"
            else:
                label="pos"
            print("\"%s\",%s"%(m.group(1),label))
    f.close()
    
