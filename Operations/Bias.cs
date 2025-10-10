using NeuralNetwork.Operations.ParamOperations;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Operations
{
    class Bias<T>: ParamOperation<T>
    {
        public Bias(Matrix2d<T> param) : base(param)
        {
        }
        public Bias(KeyValuePair<int, int> param_size, double from, double to) : base(param_size, from, to)
        {
        }

        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input) => 
            Matrix2d<T>.OperateEach(input, Param, (T a, T b) => (dynamic)a + b);

        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput) => dOutput;
        protected override Matrix2d<T> CalculateParamDeriv(Matrix2d<T> input, Matrix2d<T> dOutput) => dOutput;
    }
}
