class my_ML:
      def __init__(self,s):
      	    self.algorithms=[s]

      def add_algorithm(self,s):
            self.algorithms.append(s)
            
      def run(self):
            print("After class assignment:",self.algorithms)


ml_1=my_ML("gradient boosted trees")
ml_1.add_algorithm("decision trees")
ml_1.run()

ml_2=my_ML("bayesian belief nets")
ml_2.add_algorithm("language models")
ml_2.run()

      	  
print("ML-1:",ml_1.algorithms)
print("ML-2:",ml_2.algorithms)
