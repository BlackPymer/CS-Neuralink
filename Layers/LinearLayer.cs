using NeuralNetwork.Operations.ParamOperations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    class LinearLayer<T>: Layer<T>
    {
        public LinearLayer(int inputSize, int outputSize, double from=-1, double to=-1): 
            base(inputSize, new List<Operations.Operation<T>>
            {
                new Weights<T>(new KeyValuePair<int,int>(inputSize, outputSize),from, to),
                new Bias<T>(outputSize, from, to)
            })
        {

        }
    }
}
