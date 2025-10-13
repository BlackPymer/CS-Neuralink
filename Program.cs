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

            Operation<double>[] operations = new Operation<double>[]
            {
                new Weights<double>(new KeyValuePair<int,int>(4,3), -1, 1),
                new Bias<double>(new KeyValuePair<int,int>(1,3), -1, 1),
                new SigmoidOperation<double>()
            };

            Matrix2d<double>[] outputs = new Matrix2d<double>[3];
            for (int i = 0; i < 3; i++)
                outputs[i] = operations[i].Forward((i == 0) ? input : outputs[i - 1]);

            Console.WriteLine(outputs[2].ToString());

            Matrix2d<double>[] backwards = new Matrix2d<double>[3];
            Matrix2d<double> loss_grad = new Matrix2d<double>(1, 3);
            loss_grad.Random(0, 1);
            for (int i = 2; i >= 0; i--)
                backwards[i] = operations[i].Backward((i == 2) ?  loss_grad: backwards[i + 1]);
        }
    }
}
