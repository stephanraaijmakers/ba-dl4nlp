import sys
import re

f=open(sys.argv[1],"r")
label=sys.argv[2]

print("review,label")
for line in f:
   line=line.rstrip()
   line=re.sub("\"","",line)
   line=re.sub("\s[\.\?\!]\s","",line)
   m=re.match("(.+)\s*$",line)
   if m:
       line=m.group(1)
       print("\"%s\",%s"%(line,label))
   else:
      print("ILLEGAL:",line)
      exit(0)
f.close()


