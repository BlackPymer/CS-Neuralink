namespace NeuralNetwork.Operations
{
    abstract class SimpleOperation<T> : Operation<T>
    {

        public override Matrix2d<T> Forward(Matrix2d<T> input)
        {
            Input = input;
            Output = CalculateOutput(Input);
            return Output;
        }
        public override Matrix2d<T> Backward(Matrix2d<T> dOutput)
        {
            return CalculateDerivative(dOutput);
        }

        protected abstract Matrix2d<T> CalculateOutput(Matrix2d<T> input);
        protected abstract Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput);
    }
}
