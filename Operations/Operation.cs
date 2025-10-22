using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Operations
{
    /// <summary>
    /// Represents an abstract operation in a neural network layer.
    /// Defines the interface for forward and backward propagation.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    abstract class Operation<T>
    {
        /// <summary>
        /// Stores the input matrix used during the forward pass.
        /// Typically used later during backpropagation.
        /// </summary>
        protected Matrix2d<T> Input;

        /// <summary>
        /// Stores the output matrix produced during the forward pass.
        /// May be used for debugging or visualization.
        /// </summary>
        protected Matrix2d<T> Output;

        /// <summary>
        /// Performs the forward pass of the operation.
        /// Applies the transformation to the input and stores intermediate results.
        /// </summary>
        /// <param name="input">Input matrix to process.</param>
        /// <returns>Output matrix after applying the operation.</returns>
        public abstract Matrix2d<T> Forward(Matrix2d<T> input);

        /// <summary>
        /// Performs the backward pass of the operation.
        /// Computes the gradient of the loss with respect to the input.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        public abstract Matrix2d<T> Backward(Matrix2d<T> dOutput);
    }
}

