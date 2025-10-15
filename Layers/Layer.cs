using NeuralNetwork.Operations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    class Layer<T>
    {
        protected readonly int Size;
        protected readonly List<Operation<T>> operations;

        private KeyValuePair<int, int> _lastOutputSize;

        public Layer(int layerSize, List<Operation<T>> operations)
        {
            Size = layerSize;
            this.operations = operations;
        }

        public virtual Matrix2d<T> ForwardPropogation(Matrix2d<T> propogation)
        {
            if (propogation.Columns != Size)
                throw new LayerWrongInputSize("Wrong input size. Given: " + propogation.Columns.ToString() + ". Expected: " + Size.ToString());
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

    }
}
