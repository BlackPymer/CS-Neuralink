using NeuralNetwork.Layers;
using NeuralNetwork.Losses;
using NeuralNetwork.NeuralNetwork;
using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    /// <summary>
    /// Represents a generic neural network implementation with support for arbitrary numeric types.
    /// Provides training, testing, and inference capabilities through forward and backward propagation.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., double, float, decimal).</typeparam>
    class NeuralNetwork<T>
    {
        /// <summary>
        /// Collection of layers that compose the neural network architecture.
        /// Layers are processed sequentially during forward propagation.
        /// </summary>
        private readonly List<Layer<T>> layers;

        /// <summary>
        /// Loss function used to measure prediction accuracy and compute gradients during training.
        /// </summary>
        private readonly Loss<T> loss;

        /// <summary>
        /// Initializes a new neural network with the specified architecture and loss function.
        /// </summary>
        /// <param name="layers">List of layers defining the network architecture. Must be non-empty and properly connected.</param>
        /// <param name="loss">Loss function for measuring prediction accuracy and computing gradients.</param>
        /// <exception cref="ArgumentNullException">Thrown when layers or loss is null.</exception>
        /// <exception cref="ArgumentException">Thrown when layers list is empty.</exception>
        public NeuralNetwork(List<Layer<T>> layers, Loss<T> loss)
        {
            if (layers == null)
                throw new ArgumentNullException(nameof(layers), "Layers list cannot be null.");
            if (loss == null)
                throw new ArgumentNullException(nameof(loss), "Loss function cannot be null.");
            if (layers.Count == 0)
                throw new ArgumentException("Neural network must have at least one layer.", nameof(layers));

            this.layers = layers;
            this.loss = loss;
        }

        /// <summary>
        /// Trains the neural network using the provided training data through gradient descent optimization.
        /// Performs forward propagation, computes loss gradients, and updates network parameters.
        /// </summary>
        /// <param name="inputBatch">List of input matrices for training. Each matrix represents one training sample.</param>
        /// <param name="outputBatch">List of expected output matrices corresponding to inputBatch. Must have same count as inputBatch.</param>
        /// <param name="epoches">Number of complete passes through the training dataset.</param>
        /// <param name="applyEvery">Frequency of gradient application and loss reporting. Gradients are accumulated and applied every applyEvery samples.</param>
        /// <param name="learningRate">Step size for parameter updates during gradient descent. Controls convergence speed and stability.</param>
        /// <exception cref="BatchSizeMismatchError">Thrown when inputBatch and outputBatch have different counts.</exception>
        /// <exception cref="ArgumentNullException">Thrown when inputBatch or outputBatch is null.</exception>
        /// <exception cref="ArgumentException">Thrown when epoches or applyEvery is non-positive.</exception>
        public void Train(List<Matrix2d<T>> inputBatch, List<Matrix2d<T>> outputBatch, int epoches, int applyEvery, double learningRate = 0.01)
        {
            if (inputBatch == null)
                throw new ArgumentNullException(nameof(inputBatch), "Input batch cannot be null.");
            if (outputBatch == null)
                throw new ArgumentNullException(nameof(outputBatch), "Output batch cannot be null.");
            if (inputBatch.Count != outputBatch.Count)
                throw new BatchSizeMismatchError("Output batch size is " + outputBatch.Count.ToString() +
                    ", while input batch size is " + inputBatch.Count.ToString());
            if (epoches <= 0)
                throw new ArgumentException("Number of epochs must be positive.", nameof(epoches));
            if (applyEvery <= 0)
                throw new ArgumentException("ApplyEvery must be positive.", nameof(applyEvery));
            if (learningRate <= 0)
                throw new ArgumentException("Learning rate must be positive.", nameof(learningRate));

            for (int j = 0; j < epoches; j++)
            {
                double epochLoss = 0;
                for (int i = 0; i < inputBatch.Count; i++)
                {
                    Matrix2d<T> output = inputBatch[i];
                    foreach (var layer in layers)
                        output = layer.ForwardPropogation(output);

                    Matrix2d<T> gradient = loss.CalculateLossGradient(output, outputBatch[i]);
                    for (int k = layers.Count - 1; k >= 0; k--)
                        gradient = layers[k].BackwardPropogation(gradient, (i + (inputBatch.Count * j) + 1) % applyEvery == 0, learningRate);
                    if ((j + 1) % applyEvery == 0)
                        epochLoss += loss.CalculateLoss(output, outputBatch[i]);
                }
                if ((j + 1) % applyEvery == 0)
                {
                    Console.WriteLine($"Epoch {j + 1}: Loss = {epochLoss / inputBatch.Count}");

                }
            }
        }

        /// <summary>
        /// Evaluates the neural network's performance on a test dataset without updating parameters.
        /// Computes the average loss across all test samples to measure model accuracy.
        /// </summary>
        /// <param name="inputBatch">List of input matrices for testing. Each matrix represents one test sample.</param>
        /// <param name="outputBatch">List of expected output matrices corresponding to inputBatch. Must have same count as inputBatch.</param>
        /// <returns>Average loss value across all test samples. Lower values indicate better performance.</returns>
        /// <exception cref="BatchSizeMismatchError">Thrown when inputBatch and outputBatch have different counts.</exception>
        /// <exception cref="ArgumentNullException">Thrown when inputBatch or outputBatch is null.</exception>
        public double Test(List<Matrix2d<T>> inputBatch, List<Matrix2d<T>> outputBatch)
        {
            if (inputBatch == null)
                throw new ArgumentNullException(nameof(inputBatch), "Input batch cannot be null.");
            if (outputBatch == null)
                throw new ArgumentNullException(nameof(outputBatch), "Output batch cannot be null.");
            if (inputBatch.Count != outputBatch.Count)
                throw new BatchSizeMismatchError("Output batch size is " + outputBatch.Count.ToString() +
                    ", while input batch size is " + inputBatch.Count.ToString());

            double totalLoss = 0;
            for (int i = 0; i < inputBatch.Count; i++)
            {
                Matrix2d<T> output = inputBatch[i];
                foreach (var layer in layers)
                    output = layer.ForwardPropogation(output);
                totalLoss += loss.CalculateLoss(output, outputBatch[i]);
            }
            return totalLoss / inputBatch.Count;
        }

        /// <summary>
        /// Performs inference on a single input sample through forward propagation.
        /// Returns the network's prediction without updating any parameters.
        /// </summary>
        /// <param name="input">Input matrix representing a single sample to process.</param>
        /// <returns>Output matrix containing the network's prediction for the given input.</returns>
        /// <exception cref="ArgumentNullException">Thrown when input is null.</exception>
        /// <exception cref="LayerWrongInputSize">Thrown when input dimensions don't match the first layer's expected input size.</exception>
        public Matrix2d<T> GetOutput(Matrix2d<T> input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input), "Input cannot be null.");
            foreach (var layer in layers)
                input = layer.ForwardPropogation(input);
            return input;
        }
    }
}
