import sys
import re

def main(f):
    with open(f) as fp:
        lines=[x.rstrip() for x in fp.readlines()]
    for line in lines:
        m=re.match("^[^,]+,\"(.+)\"$",line)
        if m:
            print(m.group(1))

if __name__=="__main__":
    main(sys.argv[1])
    
