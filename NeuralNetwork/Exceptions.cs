using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.NeuralNetwork
{
    class BatchSizeMismatchError : Exception
    {
        public BatchSizeMismatchError() { }
        public BatchSizeMismatchError(string message) : base(message) { }
    }
}
