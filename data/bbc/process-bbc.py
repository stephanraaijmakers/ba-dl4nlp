import sys
import re
from glob import glob

def main(files):
    files=glob(files)
    print("label,news_item")
    for f in files:
        m=re.match("^([^\/]+)\/.+",f)
        if m:
            label=m.group(1)
        else:
            print("Error",f)
            continue
        with open(f,"r") as inp:
            lines=' '.join([line.rstrip() for line in inp.readlines() if line!=''])
            lines=re.sub("\"","\'",lines)
            lines=re.sub("  "," ",lines)
            print("%s,\"%s\""%(label,lines))
            inp.close()

if __name__=="__main__":
    files=sys.argv[1] # use \"...\"
    main(files)
    exit(0)

    
            
            
        
