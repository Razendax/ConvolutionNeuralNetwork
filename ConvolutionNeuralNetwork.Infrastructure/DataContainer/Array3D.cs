using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ConvolutionNeuralNetwork.Infrastructure.DataContainer
{
    public class Array3D : IArray
    {
        private double[,,] _data;

        private int _insertIndex = 0;

        public double this[int height, int width, int depth]
        {
            get { return _data[height, width, depth]; }
            set { _data[height, width, depth] = value; }
        }

        public double GetValue(int height, int width, int depth)
        {
            return _data[height, width, depth];
        }

        public int Height => _data.GetLength(0);

        public int Width => _data.GetLength(1);

        public int Depth => _data.GetLength(2);

        public Array3D() : this(1, 1, 1) { }
        public Array3D(int depth) : this(1, 1, depth) { }
        public Array3D(int height, int widht, int depth)
        {
            _data = new double[height, widht, depth];
        }

        public Array3D(double[,,] source, bool copy = false)
        {
            CopyData(source, copy);
        }

        public Array3D(List<double> source)
        {
            _data = new double[1, 1, source.Count];
            for (var i = 0; i < source.Count; i++)
                _data[0, 0, i] = source[i];
        }

        public Array3D(params double[] items)
            :this(1, 1, items.Length)
        {
            for (var i = 0; i < items.Length; i++)
                _data[0, 0, i] = items[i];
        }

        public void CopyData(Array3D source)
        {
            _data = new double[source.Height, source.Width, source.Depth];

            for (var i = 0; i < source.Height; i++)
                for (var j = 0; j < source.Width; j++)
                    for (var k = 0; k < source.Depth; k++)
                        _data[i, j, k] = source[i, j, k];
        }

        public void CopyData(double[,,] source, bool copy = false)
        {
            if (!copy)
            {
                _data = source;
                return;
            }

            _data = new double[source.GetLength(0), source.GetLength(1), source.GetLength(2)];

            for (var i = 0; i < source.GetLength(0); i++)
                for (var j = 0; j < source.GetLength(1); j++)
                    for (var k = 0; k < source.GetLength(2); k++)
                        _data[i, j, k] = source[i, j, k];
        }

        public void Initialize(double value)
        {
            for (var i = 0; i < Height; i++)
                for (var j = 0; j < Width; j++)
                    for (var k = 0; k < Depth; k++)
                        _data[i, j, k] = value;
        }

        public void Insert(Array2D array, int depth = -1)
        {
            if (depth >= 0)
                _insertIndex = depth;

            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Width; j++)
                    this[i, j, _insertIndex] = array[i, j];

            _insertIndex++;
        }

        public Array2D ExtractLayer(int layer)
        {
            var result = new Array2D(Height, Width);
            for (var i = 0; i < Height; i++)
                for (var j = 0; j < Width; j++)
                    result[i, j] = _data[i, j, layer];

            return result;
        }

        public void AddPadding(int top, int right, int bottom, int left)
        {
            var temp = _data;
            _data = new double[temp.GetLength(0) + top + bottom, temp.GetLength(1) + right + left, temp.GetLength(2)];

            for (var height = 0; height < temp.GetLength(0); height++)
                for (var width = 0; width < temp.GetLength(1); width++)
                    for (var depth = 0; depth < temp.GetLength(2); depth++)
                        _data[height + top, width + left, depth] = temp[height, width, depth];
        }

        public object Clone()
        {
            Array3D result = new Array3D(Height, Width, Depth);

            for (int i = 0; i < Height; i++)
                for (int j = 0; j < Width; j++)
                    for (int k = 0; k < Depth; k++)
                        result[i, j, k] = this[i, j, k];

            return result;
        }

        public static double Merge(Array3D image, Array3D filter, int height, int width)
        {
            var result = 0.0;

            for (var i = 0; i < filter.Height; i++)
                for (var j = 0; j < filter.Width; j++)
                    for (var k = 0; k < filter.Depth; k++)
                        result += image[i + height, j + width, k] * filter[i, j, k];

            return result;
        }

        public IArray Subtract(double rh)
        {
            return rh - this;
        }

        public IArray Mul(IArray lh)
        {
            return this * (lh as Array3D);
        }

        public double WeightSum()
        {
            return _data.Cast<double>().Sum();
        }

        public bool SetValue(int index, double value)
        {
            if (index >= Height * Width * Depth)
                return false;

            _data[index / (Width * Depth), index / Depth,index % Depth] = value;
            return true;
        }

        public int Count => Width * Height * Depth;

        public double this[int i]
        {
            get => _data[i / Depth / Width, i / Depth % Width, i % Depth];
            set => SetValue(i, value);
        }

        public static Array3D operator -(double lh, Array3D rh)
        {
            var result = new Array3D(rh.Height, rh.Width, rh.Depth);

            for (var i = 0; i < rh.Height; i++)
                for (var j = 0; j < rh.Width; j++)
                    for (var k = 0; k < rh.Depth; k++)
                        result[i, j, k] = lh - rh[i, j, k];

            return result;
        }

        public static Array3D operator -(Array3D lh, Array3D rh)
        {
            var result = new Array3D(lh.Height, lh.Width, lh.Depth);

            for (var i = 0; i < lh.Height; i++)
                for (var j = 0; j < lh.Width; j++)
                    for (var k = 0; k < lh.Depth; k++)
                        result[i, j, k] = lh[i, j, k] - rh[i, j, k];

            return result;
        }

        public static Array3D operator +(Array3D lh, Array3D rh)
        {
            var result = new Array3D(rh.Height, rh.Width, rh.Depth);

            for (var i = 0; i < rh.Height; i++)
                for (var j = 0; j < rh.Width; j++)
                    for (var k = 0; k < rh.Depth; k++)
                        result[i, j, k] = lh[i, j, k] + rh[i, j, k];

            return result;
        }

        public static Array3D operator *(Array3D lh, Array3D rh)
        {
            var result = new Array3D(lh.Height, lh.Width, lh.Depth);

            for (var i = 0; i < lh.Height; i++)
                for (var j = 0; j < lh.Width; j++)
                    for (var k = 0; k < lh.Depth; k++)
                        result[i, j, k] = lh[i, j, k] * rh[i, j, k];

            return result;
        }

        public static Array3D operator *(Array3D lh, double rh)
        {
            var result = new Array3D(lh.Height, lh.Width, lh.Depth);

            for (var i = 0; i < lh.Height; i++)
                for (var j = 0; j < lh.Width; j++)
                    for (var k = 0; k < lh.Depth; k++)
                        result[i, j, k] = lh[i, j, k] * rh;

            return result;
        }

        public static Array3D operator /(Array3D lh, double rh)
        {
            var result = new Array3D(lh.Height, lh.Width, lh.Depth);

            for (var i = 0; i < lh.Height; i++)
                for (var j = 0; j < lh.Width; j++)
                    for (var k = 0; k < lh.Depth; k++)
                        result[i, j, k] = lh[i, j, k] / rh;

            return result;
        }

        public static bool operator ==(Array3D rh, Array3D lh)
        {
            if ((object)rh == null || (object)lh == null)
                return false;

            if (rh.Height != lh.Height || rh.Width != lh.Width || rh.Depth != lh.Depth)
                return false;

            for (var i = 0; i < rh.Height; i++)
                for (var j = 0; j < rh.Width; j++)
                    for (var k = 0; k < rh.Depth; k++)
                        if (Math.Abs(rh[i, j, k] - lh[i, j, k]) > 0.0000001)
                            return false;

            return true;
        }

        public static bool operator !=(Array3D rh, Array3D lh)
        {
            return !(rh == lh);
        }

        public IEnumerator GetEnumerator()
        {
            for (var i = 0; i < Height; i++)
                for (var j = 0; j < Width; j++)
                    for (var k = 0; k < Depth; k++)
                        yield return _data[i, j, k];
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            foreach (var data in _data)
            {
                str.AppendFormat("{0:F3} ", data);
            }

            return str.ToString();
        }
    }
}
