using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Operations.SimpleOperations
{

    class FuncSimpleOperation<T>: SimpleOperation<T>
    {
        private readonly Func<Matrix2d<T>, Matrix2d<T>> _calculateOutput;
        private readonly Func<Matrix2d<T>, Matrix2d<T>> _calculateDerivative;

        public FuncSimpleOperation(Func<Matrix2d<T>, Matrix2d<T>> calculateOutput, Func<Matrix2d<T>, Matrix2d<T>> calculateDerivative)
        {
            _calculateOutput = calculateOutput;
            _calculateDerivative = calculateDerivative;
        }
        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input) => _calculateOutput(input);
        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput) => _calculateDerivative(dOutput);
    }
}
