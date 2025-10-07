using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Operations
{
    abstract class Operation<T>
    {
        protected Matrix2d<T> Input;
        protected Matrix2d<T> Output;
        public abstract Matrix2d<T> Forward(Matrix2d<T> input);
        public abstract Matrix2d<T> Backward(Matrix2d<T> dOutput);
    }
}
