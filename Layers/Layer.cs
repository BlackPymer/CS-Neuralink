using NeuralNetwork.Operations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Represents basic layer in neural network.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class Layer<T>
    {
        /// <summary>
        /// Stores the size of previous layer output.
        /// </summary>
        protected readonly int InputSize;

        /// <summary>
        /// Stores operations of the layer.
        /// </summary>
        protected readonly List<Operation<T>> operations;

        /// <summary>
        /// Stores the actual size of layer.
        /// Should be implemented in heritated class.
        /// </summary>
        protected int? LayerSize = null;

        /// <summary>
        /// Stores last output to except errors in Backward.
        /// </summary>
        private KeyValuePair<int, int> _lastOutputSize;

        /// <summary>
        /// Initialises layer by its input size and operations.
        /// </summary>
        /// <param name="inputSize">The size of the output of the previous layer.</param>
        /// <param name="operations">Operations that would be done in Forward in order.</param>
        public Layer(int inputSize, List<Operation<T>> operations)
        {
            InputSize = inputSize;
            this.operations = operations;
        }

        /// <summary>
        /// Does forward propogation of input through the layer.
        /// </summary>
        /// <param name="propogation">Input or previous layer forward propogation output.</param>
        /// <returns>Calculated output of the layer.</returns>
        /// <exception cref="LayerWrongInputSize">thrown when propogation size not equals to layers inputSize</exception>
        /// <exception cref="LayerOperationsMismatch">thrown when it is impossible to calculate output because of matrixes sizes mismatch.</exception>
        public virtual Matrix2d<T> ForwardPropogation(Matrix2d<T> propogation)
        {
            if (propogation.Columns != InputSize)
                throw new LayerWrongInputSize("Wrong input size. Given: " + propogation.Columns.ToString() + ". Expected: " + InputSize.ToString());
            Matrix2d<T> output = propogation;
            try
            {
                foreach (Operation<T> op in operations)
                    output = op.Forward(output);
            }
            catch (InvalidOperationException)
            {
                throw new LayerOperationsMismatch("Given matrix size is wrong");
            }
            _lastOutputSize = output.MatrixSize;
            return output;
        }

        /// <summary>
        /// Does Backward propogation through the layer.
        /// </summary>
        /// <param name="grads">The result of next layer backward propogation or loss.</param>
        /// <param name="train">If true applies grads in operations.</param>
        /// <param name="learningRate">The strength of applying grads to params.</param>
        /// <returns>Input gradients</returns>
        /// <exception cref="LayerWrongInputSize">thrown when the grads size not equals to output size.</exception>
        public virtual Matrix2d<T> BackwardPropogation(Matrix2d<T> grads, bool train = false, double learningRate=0.01)
        {
            if(grads.Rows!=_lastOutputSize.Key||grads.Columns!=_lastOutputSize.Value)
                throw new LayerWrongInputSize("Wrong input size. Given: " + grads.MatrixSize.ToString() + ". Expected: " + _lastOutputSize.ToString());
            Matrix2d<T> dOutput = grads;

            for (int i = operations.Count - 1; i >= 0; i--)
            {
                dOutput = operations[i].Backward(dOutput);
                if (train && operations[i] is ParamOperation<T>)
                    ((ParamOperation<T>)operations[i]).ApplyGradients(learningRate);
                    
            }
            return dOutput;
        }

        /// <summary>
        /// Applies gradients to Param Operations.
        /// </summary>
        /// <param name="learningRate">The strength of the gradients.</param>
        public virtual void ApplyGradients(double learningRate)
        {
            foreach(Operation<T> operation in operations)
                if(operation is ParamOperation<T> paramOperation)
                    paramOperation.ApplyGradients(learningRate);
        }

        public int? Size => LayerSize;

    }
}
