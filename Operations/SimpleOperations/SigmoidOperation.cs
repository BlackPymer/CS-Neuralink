using System;

namespace NeuralNetwork.Operations.SimpleOperations
{
    /// <summary>
    /// Represents a sigmoid activation operation applied element-wise to a matrix.
    /// The implementation is generic over T and relies on runtime conversions inside the operation body.
    /// </summary>
    /// <typeparam name="T">Element type for matrix values. The implementation performs runtime conversions; ensure your T supports the required operations at runtime.</typeparam>
    class SigmoidOperation<T> : SimpleOperation<T>
    {
        /// <summary>
        /// Applies the sigmoid function to each element of the input matrix:
        /// sigmoid(x) = 1 / (1 + exp(-x)).
        /// The method returns a new matrix containing sigmoid applied element-wise.
        /// </summary>
        /// <param name="input">Input matrix whose elements the sigmoid function will be applied to.</param>
        /// <returns>Matrix with sigmoid applied to each element of <paramref name="input"/>.</returns>
        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input)
        {
            Matrix2d<T> res = input.Copy();
            res.Operate((T val) => 1 / (1 + Math.Exp(-(dynamic)val)));
            return res;
        }

        /// <summary>
        /// Computes the backpropagated gradient for the sigmoid activation.
        /// Uses the previously stored <see cref="Output"/> (sigmoid(input)) to compute the element-wise derivative:
        /// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)), then multiplies element-wise by <paramref name="dOutput"/>.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output of this operation.</param>
        /// <returns>Gradient of the loss with respect to the input of this operation.</returns>
        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput)
        {
            Matrix2d<T> _deriv = Output.Copy();
            _deriv.Operate((T val) => val * (1 - (dynamic)val));
            return Matrix2d<T>.OperateEach(_deriv, dOutput, (T a, T b) => (dynamic)a * b);
        }
    }
}
