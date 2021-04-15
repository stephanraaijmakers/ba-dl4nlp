class my_ML:
      txt="start"
      magic_number=1234
      def __init__(self,s):
      	    self.txt = s

      def run(self):
            print("After class assignment:",self.txt)
            print("Magic number:",self.magic_number)


ml=my_ML("deep learning rules")

for i in range(10):
      ml.magic_number+=i

ml.run()

ml2=my_ML("DNN")
ml2.run()

      	  
         
