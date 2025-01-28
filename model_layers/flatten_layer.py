class FlattenLayer:
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = self.inputs.reshape((self.inputs.shape[0], self.inputs.shape[1]*self.inputs.shape[2]*self.inputs.shape[3]))
    
    def backward(self, dvalues):
        self.dinputs = dvalues.reshape((self.inputs.shape[0], self.inputs.shape[1], self.inputs.shape[2], self.inputs.shape[3]))