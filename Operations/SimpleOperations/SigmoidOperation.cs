using System;

namespace NeuralNetwork.Operations.SimpleOperations
{
    class SigmoidOperation : SimpleOperation<double>
    {
        protected override Matrix2d<double> CalculateOutput(Matrix2d<double> input)
        {
            Matrix2d<double> res = input.Copy();
            res.Operate((double val) => 1 / (1 + Math.Exp(-val)));
            return res;
        }
        protected override Matrix2d<double> CalculateDerivative(Matrix2d<double> dOutput)
        {
            Matrix2d<double> _deriv = Output.Copy();
            _deriv.Operate((double val) =>val * (1 - val));
            return Matrix2d<double>.OperateEach(_deriv, dOutput, (double a, double b) => a * b);
        }
    }
}
