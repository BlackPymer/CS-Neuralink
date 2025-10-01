using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class Matrix2d<T>
    {
        private T[,] data;
        private KeyValuePair<int, int> matrixSize;

        public Matrix2d(T[,] data)
        {
            this.data = data;
            matrixSize = new KeyValuePair<int, int>(data.GetLength(0), data.GetLength(1));
        }

        public Matrix2d(int rows, int Columns)
        {
            data = new T[rows, Columns];
            matrixSize = new KeyValuePair<int, int>(rows, Columns);
        }

        public static readonly Exception WrongMatrixSize = new InvalidOperationException("Matrix dimensions are incompatible.");

        public T this[int row, int col]
        {
            get => data[row, col];
            set => data[row, col] = value;
        }

        public int Rows => matrixSize.Key;
        public int Columns => matrixSize.Value;
        public KeyValuePair<int, int> MatrixSize => matrixSize;

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

        public static Matrix2d<T> operator +(Matrix2d<T> matrix1, Matrix2d<T> matrix2)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
                throw WrongMatrixSize;

            Matrix2d<T> result = new Matrix2d<T>(matrix1.Rows, matrix1.Columns);
            foreach (var (i, j, value) in matrix1.Elements())
                result[i, j] = (T)((dynamic)value + (dynamic)matrix2[i, j]);
            return result;
        }

        public Matrix2d<T> Transpose()
        {
            Matrix2d<T> result = new Matrix2d<T>(Columns, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    result[j, i] = data[i, j];
            return result;
        }

        public void Operate(Func<T, T> func)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    data[i, j] = func(data[i, j]);
        }

        public static Matrix2d<T> OperateEach(Matrix2d<T> matrix1, Matrix2d<T> matrix2, Func<T, T, T> func)
        {
            if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
                throw WrongMatrixSize;

            Matrix2d<T> result = new Matrix2d<T>(matrix1.Rows, matrix1.Columns);
            foreach (var (i, j, value) in matrix1.Elements())
                result[i, j] = func(value, matrix2[i, j]);

            return result;
        }

        public IEnumerable<(int i, int j, T value)> Elements()
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Columns; j++)
                    yield return (i, j, this[i, j]);
        }
        public void Random(double from, double to)
        {
            Random rnd = new Random();
            Operate(val => (T)Convert.ChangeType(from + rnd.NextDouble() * (to - from), typeof(T)));
        }

    }
}
