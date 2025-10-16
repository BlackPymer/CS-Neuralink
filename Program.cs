using NeuralNetwork.Layers;
using NeuralNetwork.Operations;
using NeuralNetwork.Operations.ParamOperations;
using NeuralNetwork.Operations.SimpleOperations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Matrix2d<double> input = new Matrix2d<double>(1, 4);
            input.Random(-3, 3);
            Console.WriteLine(input.ToString());

            List<Operation<double>> operations = new List<Operation<double>>
            {
                new Weights<double>(new KeyValuePair<int,int>(4,3), -1, 1),
                new Bias<double>(new KeyValuePair<int,int>(1,3), -1, 1),
                new SigmoidOperation<double>()
            };

            Layer<double> layer = new Layer<double>(4, operations);
            Matrix2d<double> output = layer.ForwardPropogation(input);
            Console.WriteLine(output.ToString());
            Matrix2d<double> grads = new Matrix2d<double>(1, 3);
            grads.Random(-1, 1);
            layer.BackwardPropogation(grads, true);
        }
    }
}
