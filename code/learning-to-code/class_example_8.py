class my_ML:
      def __init__(self,s):
          self.algorithms=[s]
          self.hyperparameters=[]
	    
      def add_algorithm(self,s):
          self.algorithms.append(s)

      def add_hyperparam(self,s):
          self.hyperparameters.append(s)
	    
      def run(self):
          print("After class assignment:",self.algorithms,self.hyperparameters)

class DNN_ML(my_ML):
      def __init__(self,s,nb_layers):
          my_ML.__init__(self,s)
          self.layers=nb_layers

      def run(self):
          my_ML.run(self)
          print("Layers:",self.layers)
          print("Algorithms:",self.algorithms)
	    
dnn=DNN_ML("transformer",16)
dnn.run()
