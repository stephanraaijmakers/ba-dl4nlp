import sys
import re

def ngrams(text,n=3, shift=0):
    words=text.split(" ")
    i=0
    for i in range(0,len(words)-n+1,shift):
        print(' '.join(words[i:i+n]))
    print(' '.join(words[i:]))


if __name__=="__main__":
    with open(sys.argv[1]) as f:
        lines=f.readlines()
    n=int(sys.argv[2]) # e.g. 5
    shift=int(sys.argv[3]) # e.g. 1
    for line in lines:
        m=re.match('^.*http.+',line)
        if m:
            continue
        ngrams(line.lstrip().rstrip(),n,shift)
    exit(0)
    
