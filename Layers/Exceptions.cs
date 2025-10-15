using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    class LayerOperationsMismatch : Exception
    {
        public LayerOperationsMismatch(string message) : base(message) { }

    }
    class LayerWrongInputSize : Exception
    {
        public LayerWrongInputSize(string message) : base(message) { }
    }
}
