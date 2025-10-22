using NeuralNetwork.Layers;
using NeuralNetwork.Losses;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace NeuralNetwork
{
    internal class Program
    {
        public static Matrix2d<double> RgbToCmyk(Matrix2d<double> mat)
        {
            double r, g, b;
            r = mat[0, 0];
            g = mat[0, 1];
            b = mat[0,2];
            if (r < 0 || r > 1 || g < 0 || g > 1 || b < 0 || b > 1)
                throw new ArgumentOutOfRangeException("RGB values must be in the range 0–1.");

            double k = 1 - Math.Max(r, Math.Max(g, b));

            if (k == 1)
            {
                return new Matrix2d<double>(new double[,] { { 0, 0, 0, 1 } }); // Pure black
            }

            double c = (1 - r - k) / (1 - k);
            double m = (1 - g - k) / (1 - k);
            double y = (1 - b - k) / (1 - k);

            return new Matrix2d<double>(new double[,] { { c, m, y, k } });
        }

        static KeyValuePair<List<Matrix2d<double>>,List<Matrix2d<double>>> GenerateBatch(int batchSize)
        {
            List<Matrix2d<double>> inputs = new List<Matrix2d<double>>(batchSize);
            List<Matrix2d<double>> outputs = new List<Matrix2d<double>>(batchSize);
            for(int i = 0; i < batchSize; i++)
            {
                Matrix2d<double> input = new Matrix2d<double>(1,3);
                Matrix2d<double> output = new Matrix2d<double>(1, 4);
                inputs.Add(input);
                outputs.Add(output);

                inputs[i].Random(0, 1);
                outputs[i] = RgbToCmyk(inputs[i]);
            }
            return new KeyValuePair<List<Matrix2d<double>>, List<Matrix2d<double>>>( inputs, outputs );
        }
        static void Main(string[] args)
        {

            NeuralNetwork<double> neuralNetwork = new NeuralNetwork<double>(new List<Layer<double>>
            {
                new SigmoidLayer<double>(3,64),
                new SigmoidLayer<double>(64, 4)
            }, new MSE<double>());


            var trainBatch = GenerateBatch(1024);
            neuralNetwork.Train(trainBatch.Key, trainBatch.Value, 10000, 100);
            var testBatch = GenerateBatch(128);
            Console.WriteLine(neuralNetwork.Test(testBatch.Key, testBatch.Value));


        }
    }
}
