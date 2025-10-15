using System.Collections.Generic;

namespace NeuralNetwork.Operations
{
    /// <summary>
    /// Represents an abstract parameterized operation in a neural network.
    /// Extends <see cref="Operation{T}"/> by introducing learnable parameters and their gradients.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    abstract class ParamOperation<T> : Operation<T>
    {
        public readonly int InputSize;

        /// <summary>
        /// Learnable parameter matrix (e.g., weights or biases).
        /// </summary>
        protected Matrix2d<T> Param;

        /// <summary>
        /// Gradient of the loss with respect to the parameters.
        /// Accumulated during backpropagation.
        /// </summary>
        private Matrix2d<T> DParam;

        /// <summary>
        /// Initializes the operation with a predefined parameter matrix.
        /// </summary>
        /// <param name="param">Matrix of parameters to be used in the operation.</param>
        public ParamOperation(Matrix2d<T> param)
        {
            Param = param;
            DParam = new Matrix2d<T>(param.MatrixSize);
        }

        /// <summary>
        /// Initializes the operation with a randomly generated parameter matrix.
        /// </summary>
        /// <param name="param_size">Size of the parameter matrix as (rows, columns).</param>
        /// <param name="from">Lower bound for random initialization.</param>
        /// <param name="to">Upper bound for random initialization.</param>
        public ParamOperation(KeyValuePair<int, int> param_size, double from, double to)
        {
            Param = new Matrix2d<T>(param_size);
            Param.Random(from, to);
            DParam = new Matrix2d<T>(param_size);
        }

        /// <summary>
        /// Reinitializes the parameter matrix with random values and resets its gradient.
        /// </summary>
        /// <param name="from">Lower bound for random values.</param>
        /// <param name="to">Upper bound for random values.</param>
        public void Randomise(double from, double to)
        {
            Param.Random(from, to);
            DParam.FillZero();
        }

        /// <summary>
        /// Performs the forward pass of the operation.
        /// Stores the input and computes the output using the current parameters.
        /// </summary>
        /// <param name="input">Input matrix to process.</param>
        /// <returns>Output matrix after applying the operation.</returns>
        public override Matrix2d<T> Forward(Matrix2d<T> input)
        {
            Input = input;
            Output = CalculateOutput(input);
            return Output;
        }

        /// <summary>
        /// Performs the backward pass of the operation.
        /// Accumulates the gradient with respect to the parameters and returns the gradient with respect to the input.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        public override Matrix2d<T> Backward(Matrix2d<T> dOutput)
        {
            DParam += CalculateParamDeriv(Input, dOutput);
            return CalculateDerivative(dOutput);
        }

        /// <summary>
        /// Resets the accumulated gradient of the parameters to zero.
        /// </summary>
        public void ResetGrad()
        {
            DParam.FillZero();
        }

        /// <summary>
        /// Applies the accumulated gradients to the parameters using the specified learning rate.
        /// After application, the gradient is reset.
        /// </summary>
        /// <param name="learningRate">Learning rate used to scale the gradient update.</param>
        public void ApplyGradients(double learningRate)
        {
            DParam.Operate(v => (T)((dynamic)v * learningRate));
            Param -= DParam;
            DParam.FillZero();
        }

        /// <summary>
        /// Computes the output of the operation during the forward pass.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="input">Input matrix.</param>
        /// <returns>Output matrix.</returns>
        protected abstract Matrix2d<T> CalculateOutput(Matrix2d<T> input);

        /// <summary>
        /// Computes the gradient of the loss with respect to the parameters.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="input">Input matrix used during the forward pass.</param>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the parameters.</returns>
        protected abstract Matrix2d<T> CalculateParamDeriv(Matrix2d<T> input, Matrix2d<T> dOutput);

        /// <summary>
        /// Computes the gradient of the loss with respect to the input.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        protected abstract Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput);
    }
}
