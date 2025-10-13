using System.Collections.Generic;

namespace NeuralNetwork.Operations
{
    /// <summary>
    /// Represents a bias operation in a neural network layer.
    /// Adds learnable bias parameters to each element of the input.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class Bias<T> : ParamOperation<T>
    {
        /// <summary>
        /// Initializes the bias operation with a predefined parameter matrix.
        /// </summary>
        /// <param name="param">Matrix of bias values to be added to the input.</param>
        public Bias(Matrix2d<T> param) : base(param)
        {
        }

        /// <summary>
        /// Initializes the bias operation with a randomly generated parameter matrix.
        /// </summary>
        /// <param name="param_size">Size of the bias matrix as (rows, columns).</param>
        /// <param name="from">Lower bound for random initialization.</param>
        /// <param name="to">Upper bound for random initialization.</param>
        public Bias(KeyValuePair<int, int> param_size, double from, double to) : base(param_size, from, to)
        {
        }

        /// <summary>
        /// Computes the forward pass by adding the bias matrix to the input element-wise.
        /// </summary>
        /// <param name="input">Input matrix to which bias is added.</param>
        /// <returns>Output matrix after bias addition.</returns>
        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input) =>
            Matrix2d<T>.OperateEach(input, Param, (T a, T b) => (dynamic)a + b);

        /// <summary>
        /// Computes the gradient of the loss with respect to the input.
        /// Since bias addition does not affect the gradient shape, it is passed through unchanged.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput) => dOutput;

        /// <summary>
        /// Computes the gradient of the loss with respect to the bias parameters.
        /// </summary>
        /// <param name="input">Input matrix used during the forward pass.</param>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the bias parameters.</returns>
        protected override Matrix2d<T> CalculateParamDeriv(Matrix2d<T> input, Matrix2d<T> dOutput) => dOutput;
    }
}
