using NeuralNetwork.Operations.ParamOperations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Represents a linear regression layer.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class LinearLayer<T>: Layer<T>
    {
        /// <summary>
        /// Initialises the layer and its operations.
        /// </summary>
        /// <param name="inputSize">The size of the output of the previous layer.</param>
        /// <param name="outputSize">The size of the output of this layer.</param>
        /// <param name="from">Randomises weights and bias from.</param>
        /// <param name="to">Randomises weights and bias to.</param>
        public LinearLayer(int inputSize, int outputSize, double from=-1, double to=-1): 
            base(inputSize, new List<Operations.Operation<T>>
            {
                new Weights<T>(new KeyValuePair<int,int>(inputSize, outputSize),from, to),
                new Bias<T>(outputSize, from, to)
            })
        {
            LayerSize = outputSize;
        }
    }
}
