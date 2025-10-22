using System;

namespace NeuralNetwork.Losses
{
    /// <summary>
    /// Represents Mean Squared Error loss class.
    /// </summary>
    /// <typeparam name="T">Numeric type used for computations (e.g., float, double).</typeparam>
    class MSE<T> : Loss<T>
    {
        public override dynamic CalculateLoss(Matrix2d<T> prediction, Matrix2d<T> correctResult)
        {
            return Matrix2d<T>.OperateEach(prediction, correctResult, (T val1, T val2) =>
            Math.Pow((dynamic)val1 - val2, 2)).Sum() / (prediction.Rows * prediction.Columns);
        }
        public override Matrix2d<T> CalculateLossGradient(Matrix2d<T> prediction, Matrix2d<T> correctResult)
        {
            return Matrix2d<T>.OperateEach(prediction, correctResult, (T val1, T val2) => 2 * ((dynamic)val1 - val2));
        }
    }
}
