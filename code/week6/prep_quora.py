import re
import sys

# Topic_Id | Topic_Name | Sent_1 | Sent_2 | Label | Sent_1_tag | Sent_2_tag |
#The "Label" column for *dev/train data * is in a format like "(1, 4)", which means among 5 votes from Amazon Mechanical turkers only 1 is positive and 4 are negative. We would suggest map them to binary labels as follows:
#paraphrases: (3, 2) (4, 1) (5, 0)
#non-paraphrases: (1, 4) (0, 5)
#debatable: (2, 3)  which you may discard if training binary classifier
# https://github.com/cocoxu/SemEval-PIT2015



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
