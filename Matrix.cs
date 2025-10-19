using System;
using System.Collections.Generic;
using System.Threading;

namespace NeuralNetwork
{
    /// <summary>
    /// Represents a two-dimensional matrix with generic element type.
    /// Provides basic matrix operations used by the neural network.
    /// </summary>
    /// <typeparam name="T">Numeric type used for matrix elements (e.g., double, float, int).</typeparam>
    class Matrix2d<T>
    {
        private T[,] data;
        private KeyValuePair<int, int> matrixSize;

        /// <summary>
        /// Initializes matrix from an existing 2D array. The array is copied to avoid external mutations.
        /// </summary>
        /// <param name="data">Source 2D array.</param>
        public Matrix2d(T[,] data)
        {
            var rows = data.GetLength(0);
            var cols = data.GetLength(1);
            this.data = new T[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    this.data[i, j] = data[i, j];
            matrixSize = new KeyValuePair<int, int>(rows, cols);
        }

        /// <summary>
        /// Initializes an empty matrix with given dimensions.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="columns">Number of columns.</param>
        public Matrix2d(int rows, int columns)
        {
            data = new T[rows, columns];
            matrixSize = new KeyValuePair<int, int>(rows, columns);
        }

        /// <summary>
        /// Initializes an empty matrix with given size pair (rows, columns).
        /// </summary>
        /// <param name="size">Pair where Key = rows, Value = columns.</param>
        public Matrix2d(KeyValuePair<int, int> size)
        {
            data = new T[size.Key, size.Value];
            matrixSize = size;
        }

        /// <summary>
        /// Creates a deep copy of the matrix.
        /// </summary>
        /// <returns>New Matrix2d instance with copied data.</returns>
        public Matrix2d<T> Copy()
        {
            var clone = new T[Rows, Columns];
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    clone[i, j] = data[i, j];
            return new Matrix2d<T>(clone);
        }

        /// <summary>
        /// Exception thrown when matrix dimensions are incompatible for an operation.
        /// </summary>
        public static readonly Exception WrongMatrixSize = new InvalidOperationException("Matrix dimensions are incompatible.");

        /// <summary>
        /// Gets or sets the element at specified row and column.
        /// </summary>
        public T this[int row, int col]
        {
            get => data[row, col];
            set => data[row, col] = value;
        }

        /// <summary>
        /// Number of rows.
        /// </summary>
        public int Rows => matrixSize.Key;

        /// <summary>
        /// Number of columns.
        /// </summary>
        public int Columns => matrixSize.Value;

        /// <summary>
        /// Size pair (rows, columns).
        /// </summary>
        public KeyValuePair<int, int> MatrixSize => matrixSize;

        /// <summary>
        /// Matrix multiplication (matrix1 * matrix2). Throws if inner dimensions mismatch.
        /// Note: uses dynamic arithmetic for generic T; consider replacing with numeric provider for performance and type safety.
        /// </summary>
        public static Matrix2d<T> operator *(Matrix2d<T> matrix1, Matrix2d<T> matrix2)
        {
            if (matrix1.Columns != matrix2.Rows)
                throw WrongMatrixSize;

            Matrix2d<T> result = new Matrix2d<T>(matrix1.Rows, matrix2.Columns);

            for (int i = 0; i < matrix1.Rows; i++)
            {
                for (int j = 0; j < matrix2.Columns; j++)
                {
                    dynamic sum = default(T);
                    for (int k = 0; k < matrix1.Columns; k++)
                        sum += (dynamic)matrix1[i, k] * (dynamic)matrix2[k, j];
                    result[i, j] = (T)sum;
                }
            }
            return result;
        }

        /// <summary>
        /// Element-wise matrix addition. Throws if sizes differ.
        /// </summary>
        public static Matrix2d<T> operator +(Matrix2d<T> matrix1, Matrix2d<T> matrix2)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
                throw WrongMatrixSize;

            Matrix2d<T> result = new Matrix2d<T>(matrix1.MatrixSize);
            foreach (var (i, j, value) in matrix1.Elements())
                result[i, j] = (T)((dynamic)value + (dynamic)matrix2[i, j]);
            return result;
        }

        /// <summary>
        /// Element-wise matrix subtraction. Returns new matrix; does not mutate operands.
        /// </summary>
        public static Matrix2d<T> operator -(Matrix2d<T> matrix1, Matrix2d<T> matrix2)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
                throw WrongMatrixSize;

            Matrix2d<T> result = new Matrix2d<T>(matrix1.MatrixSize);
            foreach (var (i, j, value) in matrix1.Elements())
                result[i, j] = (T)((dynamic)value - (dynamic)matrix2[i, j]);
            return result;
        }

        /// <summary>
        /// Returns the transposed matrix.
        /// </summary>
        public Matrix2d<T> Transpose()
        {
            Matrix2d<T> result = new Matrix2d<T>(Columns, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    result[j, i] = data[i, j];
            return result;
        }

        /// <summary>
        /// Applies a unary function to each element of the matrix in-place.
        /// </summary>
        /// <param name="func">Function to apply to each element.</param>
        public void Operate(Func<T, T> func)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    data[i, j] = func(data[i, j]);
        }

        /// <summary>
        /// Fills the matrix with the default value of T (commonly zero for numeric types).
        /// </summary>
        public void FillZero()
        {
            Operate(_ => default(T));
        }

        /// <summary>
        /// Fills the matrix with ones. Uses Convert.ChangeType and may fail for non-numeric T.
        /// Consider replacing with a numeric provider for robust behavior.
        /// </summary>
        public void FillOnes()
        {
            Operate(_ => (T)Convert.ChangeType(1, typeof(T)));
        }

        /// <summary>
        /// Fills the matrix with a specified value.
        /// </summary>
        /// <param name="value">Value to fill.</param>
        public void Fill(T value)
        {
            Operate(_ => value);
        }

        /// <summary>
        /// Applies a binary function to corresponding elements of two matrices and returns the result.
        /// Throws if sizes differ.
        /// </summary>
        public static Matrix2d<T> OperateEach(Matrix2d<T> matrix1, Matrix2d<T> matrix2, Func<T, T, T> func)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
                throw WrongMatrixSize;

            Matrix2d<T> result = new Matrix2d<T>(matrix1.MatrixSize);
            foreach (var (i, j, value) in matrix1.Elements())
                result[i, j] = func(value, matrix2[i, j]);

            return result;
        }

        /// <summary>
        /// Enumerates elements as tuples (row, column, value).
        /// </summary>
        public IEnumerable<(int i, int j, T value)> Elements()
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    yield return (i, j, this[i, j]);
        }

        private static readonly ThreadLocal<Random> rnd = new ThreadLocal<Random>(() => new Random());

        /// <summary>
        /// Fills the matrix with random values in [from, to). Uses Convert.ChangeType; consider numeric provider for type safety.
        /// </summary>
        /// <param name="from">Inclusive lower bound.</param>
        /// <param name="to">Exclusive upper bound.</param>
        public void Random(double from, double to)
        {
            Operate(_ => (T)Convert.ChangeType(from + rnd.Value.NextDouble() * (to - from), typeof(T)));
        }

        /// <summary>
        /// Converts matrix to print format.
        /// </summary>
        public override string ToString()
        {
            string res = "{\n";
            for (int i = 0; i < Rows; i++)
            {
                res += "\t{ ";
                for (int j = 0; j < Columns; j++)
                    res += data[i, j].ToString() + " ";
                res += "}\n";
            }
            return res + "}";
        }

        public dynamic Sum()
        {
            dynamic result = 0;
            foreach (var (i, j, value) in Elements())
                result += data[i, j];
            return result;
        }
    }
}
