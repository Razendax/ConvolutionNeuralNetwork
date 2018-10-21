

using System.Collections;
using System.IO;
using System.Text;

namespace ConvolutionNeuralNetwork.Infrastructure.DataContainer
{
    public class Array2D : IArray
    {
        private double[,] _data;

        public double this[int height, int width]
        {
            get => _data[height, width];
            set => _data[height, width] = value;
        }

        public int Height => _data.GetLength(0);
        public int Width => _data.GetLength(1);

        public Array2D() : this(0, 0) { }

        public Array2D(int height, int widht)
        {
            _data = new double[height, widht];
        }

        public object Clone()
        {
            Array2D result = new Array2D(Height, Width);
            for (var i = 0; i < Height; i++)
            for (var j = 0; j < Width; j++)
                result[i, j] = this[i, j];

            return result;
        }

        public IArray Subtract(double lh)
        {
            return lh - this;
        }

        public IArray Mul(IArray rh)
        {
            return this * (rh as Array2D);
        }

        public bool SetValue(int index, double value)
        {
            if (index >= Width * Height)
                return false;

            _data[index / Width, index % Width] = value;
            return true;
        }

        public int Count => Width * Height;

        public double this[int i]
        {
            get => _data[i /Width, i % Width];
            set => SetValue(i, value);
        }

        public static Array2D operator -(double lh, Array2D rh)
        {
            var result = new Array2D(rh.Height, rh.Width);

            for (var i = 0; i < rh.Height; i++)
            for (var j = 0; j < rh.Width; j++)
                result[i, j] = lh - rh[i, j];

            return result;
        }

        public static Array2D operator *(Array2D lh, Array2D rh)
        {
            var result = new Array2D(rh.Height, rh.Width);

            for (var i = 0; i < rh.Height; i++)
            for (var j = 0; j < rh.Width; j++)
                result[i, j] = lh[i, j] * rh[i, j];

            return result;
        }

        public IEnumerator GetEnumerator()
        {
            for (var i = 0; i < Height; i++)
                for (var j = 0; j < Width; j++)
                    yield return _data[i, j];
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            foreach (var data in _data)
            {
                str.AppendFormat("{0} ", data);
            }

            return str.ToString();
        }
    }
}
