using System;
using System.Collections.Generic;

namespace NeuralNetwork.Operations
{
    abstract class ParamOperation<T> : Operation<T>
    {
        protected Matrix2d<T> Param;
        private Matrix2d<T> DParam;
        public ParamOperation(Matrix2d<T> param)
        {
            Param = param;
            DParam = new Matrix2d<T>(param.MatrixSize);
        }
        public ParamOperation(KeyValuePair<int, int> param_size, double from, double to)
        {
            Param = new Matrix2d<T>(param_size);
            Param.Random(from, to);
            DParam = new Matrix2d<T>(param_size);
        }
        public void Randomise(double from, double to)
        {
            Param.Random(from, to);
            DParam.FillZero();
        }
        public override Matrix2d<T> Forward(Matrix2d<T> input)
        {
            Input = input;
            Output = CalculateOutput(input);
            return Output;
        }
        public override Matrix2d<T> Backward(Matrix2d<T> dOutput)
        {
            DParam +=CalculateParamDeriv(Input, dOutput);
            return CalculateDerivative(dOutput);
        }

        public void ResetGrad()
        {
            DParam.FillZero();
        }

        public void ApplyGradients(double learningRate)
        {
            DParam.Operate((T val) => val = (T)Convert.ChangeType((dynamic)val * learningRate, typeof(T)));
            Param += DParam;
            DParam.FillZero();
        }

        protected abstract Matrix2d<T> CalculateOutput(Matrix2d<T> input);
        protected abstract Matrix2d<T> CalculateParamDeriv(Matrix2d<T> input,Matrix2d<T> dOutput);
        protected abstract Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput);
    }
}