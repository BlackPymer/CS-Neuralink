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

            Layer<double> layer = new LinearLayer<double>(4, 3);
            Matrix2d<double> output = layer.ForwardPropogation(input);
            Console.WriteLine(output.ToString());

            Layer<double> sigmoidLayer = new SigmoidLayer<double>(3, 5);
            Matrix2d<double> res = sigmoidLayer.ForwardPropogation(output);
            Console.WriteLine(res.ToString());

            Matrix2d<double> grads = new Matrix2d<double>(1, 5);
            grads.Random(-1, 1);
            grads = sigmoidLayer.BackwardPropogation(grads, true);
            Console.WriteLine(grads.ToString());
            layer.BackwardPropogation(grads, true);
        }
    }
}
