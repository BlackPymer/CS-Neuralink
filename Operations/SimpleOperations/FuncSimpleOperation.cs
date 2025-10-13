using System;

namespace NeuralNetwork.Operations.SimpleOperations
{
    /// <summary>
    /// Represents a simple operation defined by user-provided functions.
    /// Allows flexible definition of forward and backward behavior without subclassing.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class FuncSimpleOperation<T> : SimpleOperation<T>
    {
        /// <summary>
        /// Delegate that defines the forward computation logic.
        /// </summary>
        private readonly Func<Matrix2d<T>, Matrix2d<T>> _calculateOutput;

        /// <summary>
        /// Delegate that defines the backward computation logic (gradient with respect to input).
        /// </summary>
        private readonly Func<Matrix2d<T>, Matrix2d<T>> _calculateDerivative;

        /// <summary>
        /// Initializes a new instance of <see cref="FuncSimpleOperation{T}"/> using the provided forward and backward functions.
        /// </summary>
        /// <param name="calculateOutput">Function that computes the output from the input.</param>
        /// <param name="calculateDerivative">Function that computes the gradient with respect to the input.</param>
        public FuncSimpleOperation(
            Func<Matrix2d<T>, Matrix2d<T>> calculateOutput,
            Func<Matrix2d<T>, Matrix2d<T>> calculateDerivative)
        {
            _calculateOutput = calculateOutput;
            _calculateDerivative = calculateDerivative;
        }

        /// <summary>
        /// Computes the output of the operation using the provided delegate.
        /// </summary>
        /// <param name="input">Input matrix.</param>
        /// <returns>Output matrix.</returns>
        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input) => _calculateOutput(input);

        /// <summary>
        /// Computes the gradient of the loss with respect to the input using the provided delegate.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput) => _calculateDerivative(dOutput);
    }
}
