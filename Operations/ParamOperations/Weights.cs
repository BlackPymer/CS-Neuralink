using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Operations.ParamOperations
{
    /// <summary>
    /// Represents a weight operation in a neural network layer.
    /// Applies a linear transformation to the input using learnable parameters (weights).
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class Weights<T> : ParamOperation<T>
    {
        /// <summary>
        /// Initializes the weight operation with a predefined parameter matrix.
        /// </summary>
        /// <param name="param">Matrix of weights to be used in the operation.</param>
        public Weights(Matrix2d<T> param) : base(param)
        {
        }

        /// <summary>
        /// Initializes the weight operation with a randomly generated parameter matrix.
        /// </summary>
        /// <param name="param_size">Size of the weight matrix as (rows, columns).</param>
        /// <param name="from">Lower bound for random initialization.</param>
        /// <param name="to">Upper bound for random initialization.</param>
        public Weights(KeyValuePair<int, int> param_size, double from, double to) : base(param_size, from, to)
        {
        }

        /// <summary>
        /// Computes the forward pass by applying the weight matrix to the input.
        /// </summary>
        /// <param name="input">Input matrix to transform.</param>
        /// <returns>Transformed output matrix.</returns>
        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input)
        {
            return input * Param;
        }

        /// <summary>
        /// Computes the gradient of the loss with respect to the input.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput)
        {
            return dOutput * Param.Transpose();
        }

        /// <summary>
        /// Computes the gradient of the loss with respect to the weight parameters.
        /// </summary>
        /// <param name="input">Input matrix used during the forward pass.</param>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the parameters.</returns>
        protected override Matrix2d<T> CalculateParamDeriv(Matrix2d<T> input, Matrix2d<T> dOutput)
        {
            return input.Transpose() * dOutput;
        }
    }
}

