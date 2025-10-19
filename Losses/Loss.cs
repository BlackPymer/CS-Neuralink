using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Losses
{
    /// <summary>
    /// Represents an abstract Loss in Neural Network.
    /// Defined the interface for loss and gradient loss calculations.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    abstract class Loss<T>
    {
        /// <summary>
        /// Calculates the loss value.
        /// </summary>
        /// <param name="prediction">Matrix of predicted values.</param>
        /// <param name="correctResult">Matrix of expected result.</param>
        /// <returns></returns>
        public abstract dynamic CalculateLoss(Matrix2d<T> prediction, Matrix2d<T> correctResult);
        /// <summary>
        /// Calculates the loss gradient for every param.
        /// </summary>
        /// <param name="prediction">Matrix of predicted values.</param>
        /// <param name="correctResult">Matrix of expected result.</param>
        /// <returns></returns>
        public abstract Matrix2d<T> CalculateLossGradient(Matrix2d<T> prediction, Matrix2d<T> correctResult);
    }
}
