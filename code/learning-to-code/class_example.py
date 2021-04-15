class my_ML:

      txt="start"
      
      def ___init___(self,s):
      	    self.txt = s

      def run(self,s):
            self.set_var(s)
            print("After class assignment:",self.txt)


ml=my_ML()

ml.run("deep learning rules")

print("Final value:",ml.txt)
            
      	  
         
