namespace NeuralNetwork.Operations
{
    /// <summary>
    /// Represents a stateless operation in a neural network layer.
    /// Defines the interface for forward and backward propagation without learnable parameters.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    abstract class SimpleOperation<T> : Operation<T>
    {
        /// <summary>
        /// Performs the forward pass of the operation.
        /// Stores the input and computes the output.
        /// </summary>
        /// <param name="input">Input matrix to process.</param>
        /// <returns>Output matrix after applying the operation.</returns>
        public override Matrix2d<T> Forward(Matrix2d<T> input)
        {
            Input = input;
            Output = CalculateOutput(Input);
            return Output;
        }

        /// <summary>
        /// Performs the backward pass of the operation.
        /// Computes the gradient of the loss with respect to the input.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        public override Matrix2d<T> Backward(Matrix2d<T> dOutput)
        {
            return CalculateDerivative(dOutput);
        }

        /// <summary>
        /// Computes the output of the operation during the forward pass.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="input">Input matrix.</param>
        /// <returns>Output matrix.</returns>
        protected abstract Matrix2d<T> CalculateOutput(Matrix2d<T> input);

        /// <summary>
        /// Computes the gradient of the loss with respect to the input.
        /// Must be implemented by derived classes.
        /// </summary>
        /// <param name="dOutput">Gradient of the loss with respect to the output.</param>
        /// <returns>Gradient of the loss with respect to the input.</returns>
        protected abstract Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput);
    }
}
