using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Operations
{
    class SimpleOperation<T>: Operation<T>
    {
        private readonly Func<T, T> _forwardOperation;
        private readonly Func<T, T> _backwardOperation;
        public SimpleOperation(Func<T,T> forward, Func<T,T> backward) { 
            _forwardOperation = forward;
            _backwardOperation = backward;
        }

        public override Matrix2d<T> Forward(Matrix2d<T> input)
        {
            Input = input;
            Output = Input.Copy();
            Output.Operate(_forwardOperation);
            return Output;
        }
        public override Matrix2d<T> Backward(Matrix2d<T> dOutput) {
            Matrix2d<T> dInput = Input.Copy();
            dInput.Operate(_backwardOperation);
            return Matrix2d<T>.OperateEach(dInput, dOutput, (T inp, T grad) => (dynamic)inp * grad);
        }
    }
}
