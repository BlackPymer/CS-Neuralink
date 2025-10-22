using System;
using System.Collections.Generic;

namespace NeuralNetwork.Operations.ParamOperations
{
    /// <summary>
    /// A convenient wrapper of ParamOperation that allows behaviour to be supplied via delegates.
    /// </summary>
    /// <typeparam name="T">Numeric element type of the matrices.</typeparam>
    class FuncParamOperation<T> : ParamOperation<T>
    {
        protected readonly Func<Matrix2d<T>, Matrix2d<T>> _calculateOutput;
        protected readonly Func<Matrix2d<T>, Matrix2d<T>, Matrix2d<T>> _calculateParamDeriv;
        protected readonly Func<Matrix2d<T>, Matrix2d<T>, Matrix2d<T>> _calculateDerivative;

        /// <summary>
        /// Creates an operation with pre-initialized parameters.
        /// </summary>
        /// <param name="param">Parameter matrix (for example weights or biases). Dimensions must match what the delegates expect.</param>
        /// <param name="calculateOutput">
        /// Delegate that computes the forward pass. Accepts <c>input</c> and returns <c>output</c>.
        /// The returned matrix must have dimensions compatible with the rest of the network.
        /// </param>
        /// <param name="calculateParamDeriv">
        /// Delegate that computes the parameter gradients. Accepts <c>input</c> and <c>dOutput</c> and returns the gradient matrix for parameters.
        /// The returned matrix is expected to have the same shape as <c>param</c>.
        /// </param>
        /// <param name="calculateDerivative">
        /// Delegate that computes the input derivative. Accepts <c>param</c> and <c>dOutput</c> and returns <c>dInput</c>.
        /// Argument order is <c>(param, dOutput)</c>. The returned matrix must have the same shape as the original <c>input</c>.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown if any delegate or <c>param</c> is null.</exception>
        public FuncParamOperation(
            Matrix2d<T> param,
            Func<Matrix2d<T>, Matrix2d<T>> calculateOutput,
            Func<Matrix2d<T>, Matrix2d<T>, Matrix2d<T>> calculateParamDeriv,
            Func<Matrix2d<T>, Matrix2d<T>, Matrix2d<T>> calculateDerivative) : base(param)
        {
            _calculateOutput = calculateOutput ?? throw new ArgumentNullException(nameof(calculateOutput));
            _calculateParamDeriv = calculateParamDeriv ?? throw new ArgumentNullException(nameof(calculateParamDeriv));
            _calculateDerivative = calculateDerivative ?? throw new ArgumentNullException(nameof(calculateDerivative));
        }

        /// <summary>
        /// Creates an operation and initializes parameters randomly in the range [from, to].
        /// </summary>
        /// <param name="param_size">Pair (rows, cols) specifying the parameter matrix dimensions.</param>
        /// <param name="from">Lower bound for random initialization (inclusive).</param>
        /// <param name="to">Upper bound for random initialization (inclusive).</param>
        /// <param name="calculateOutput">
        /// Delegate that computes the forward pass. Accepts <c>input</c> and returns <c>output</c>.
        /// </param>
        /// <param name="calculateParamDeriv">
        /// Delegate that computes the parameter gradients. Accepts <c>input</c> and <c>dOutput</c> and returns the gradient matrix for parameters.
        /// </param>
        /// <param name="calculateDerivative">
        /// Delegate that computes the input derivative. Accepts <c>param</c> and <c>dOutput</c> and returns <c>dInput</c>.
        /// Argument order is <c>(param, dOutput)</c>.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown if any delegate is null.</exception>
        public FuncParamOperation(
            KeyValuePair<int, int> param_size,
            double from,
            double to,
            Func<Matrix2d<T>, Matrix2d<T>> calculateOutput,
            Func<Matrix2d<T>, Matrix2d<T>, Matrix2d<T>> calculateParamDeriv,
            Func<Matrix2d<T>, Matrix2d<T>, Matrix2d<T>> calculateDerivative) : base(param_size, from, to)
        {
            _calculateOutput = calculateOutput ?? throw new ArgumentNullException(nameof(calculateOutput));
            _calculateParamDeriv = calculateParamDeriv ?? throw new ArgumentNullException(nameof(calculateParamDeriv));
            _calculateDerivative = calculateDerivative ?? throw new ArgumentNullException(nameof(calculateDerivative));
        }

        protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input) => _calculateOutput(input);

        protected override Matrix2d<T> CalculateParamDeriv(Matrix2d<T> input, Matrix2d<T> dOutput) => _calculateParamDeriv(input, dOutput);

        protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput) => _calculateDerivative(Param, dOutput);
    }
}