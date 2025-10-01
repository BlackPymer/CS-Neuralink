using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Operation<T>
    {
        private Func<T, T> forwardOperation;
        private Func<T, T> backwardOperation;
        public Operation(Func<T,T> forward, Func<T,T> backward) { 
            forwardOperation = forward;
            backwardOperation = backward;
        }

        public Matrix2d<T> forward(Matrix2d<T> input)
        {
            input.Operate(forwardOperation);
            return input;
        }
        public Matrix2d<T> backward(Matrix2d<T> output) {
            output.Operate(backwardOperation);
            return output;
        }
    }
}
