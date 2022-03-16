import re
import sys


def jaccard_sim(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

if __name__=="__main__":
    with open(sys.argv[1],"r") as fp:
        lines=fp.readlines()
    for line in lines[1:]:
        m=re.match("^\"[0-9]+\",\"[0-9]+\",\"[0-9]+\",\"([^\"]+)\",\"([^\"]+)\",([0-9]+).*$",line.rstrip())
        if m:
            left=m.group(1)
            right=m.group(2)
            w_left=left.split(" ")
            w_right=right.split(" ")
            js=jaccard_sim(w_left, w_right)
            if js > 0.6:
                label=1
            else:
                label=0
            print("%s\t%s\t%d"%(m.group(1),m.group(2),label))
    fp.close()
    exit(0)
