with open("mydata.txt","r") as my_file:
    lines=my_file.readlines()
my_file.close()

with open("myresults.txt","w") as my_file:
    my_file.write("F1=99.9!\n")
my_file.close()


my_file=open("mydata.txt","r")
lines=my_file.readlines()
my_file.close()

