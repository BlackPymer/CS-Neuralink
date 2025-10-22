using NeuralNetwork.Layers;
using NeuralNetwork.Losses;
using NeuralNetwork.NeuralNetwork;
using System;
using System.Collections.Generic;
namespace NeuralNetwork
{
    class NeuralNetwork<T>
    {
        private readonly List<Layer<T>> layers;
        private readonly Loss<T> loss;
        public NeuralNetwork(List<Layer<T>> layers, Loss<T> loss)
        {
            this.layers = layers;
            this.loss = loss;
        }

        public void Train(List<Matrix2d<T>> inputBatch, List<Matrix2d<T>> outputBatch, int epoches, int applyEvery, double learningRate = 0.01)
        {
            if (inputBatch.Count != outputBatch.Count)
                throw new BatchSizeMismatchError("Output batch size is " + outputBatch.Count.ToString() +
                    ", while input batch size is " + inputBatch.Count.ToString());

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

        public double Test(List<Matrix2d<T>> inputBatch, List<Matrix2d<T>> outputBatch)
        {
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

        public Matrix2d<T> GetOutput(Matrix2d<T> input)
        {
            foreach (var layer in layers)
                input = layer.ForwardPropogation(input);
            return input;
        }
    }
}
