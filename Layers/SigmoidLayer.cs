using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NeuralNetwork.Operations;
using NeuralNetwork.Operations.ParamOperations;
using NeuralNetwork.Operations.SimpleOperations;
namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Respresent the Layer with sigmoid operation
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class SigmoidLayer<T>: Layer<T>
    {
        /// <summary>
        /// Initialises a layer with Sigmoid only operation.
        /// </summary>
        /// <param name="inputSize">The size of the output of the previous layer. Equals output of this layer in this case.</param>
        public SigmoidLayer(int inputSize):
            base(inputSize, 
                new List<Operation<T>>
                {
                    new SigmoidOperation<T>()
                })
        {
            LayerSize = inputSize;
        }

        /// <summary>
        /// Initialises a layer with Weights, Bias and Sigmoid operations.
        /// </summary>
        /// <param name="inputSize">The size of the output of the previous layer.</param>
        /// <param name="outputSize">The size of the output of this layer.</param>
        /// <param name="from">Randomises weights and bias from.</param>
        /// <param name="to">Randomises weights and bias to.</param>
        public SigmoidLayer(int inputSize, int outputSize, double from=-1, double to = 1):
            base(inputSize,
                new List<Operation<T>>
                {
                    new Weights<T>(new KeyValuePair<int,int>(inputSize, outputSize),from,to),
                    new Bias<T>(outputSize, from, to),
                    new SigmoidOperation<T>()
                })
        {
            LayerSize = outputSize;
        }
    }
}
