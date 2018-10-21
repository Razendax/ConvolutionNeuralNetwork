using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace ConvolutionNeuralNetwork.Infrastructure.DataContainer
{
    public class Array : IList<double>, IArray
    {
        private List<double> _data;

        public Array()
        {
            _data = new List<double>();
        }

        public Array(int capacity, bool resolve = false)
        {
            _data = new List<double>(capacity);
            if (!resolve)
                return;

            for (var i = 0; i < capacity; i++)
                _data.Add(0);
        }

        public Array(IList<double> data)
        {
            _data = new List<double>(data.Count);
            var temp = new double[data.Count];
            data.CopyTo(temp, 0);
            _data.AddRange(temp);
        }

        public double this[int index]
        {
            get => _data[index];
            set => _data[index] = value;
        }

        public int Count => _data.Count;

        public bool IsReadOnly => false;

        public void Add(double item)
        {
            _data.Add(item);
        }

        public void Clear()
        {
            _data.Clear();
        }

        public bool Contains(double item)
        {
            return _data.Contains(item);
        }

        public void CopyTo(double[] array, int arrayIndex)
        {
            _data.CopyTo(array, arrayIndex);
        }

        public IEnumerator<double> GetEnumerator()
        {
            return _data.GetEnumerator();
        }

        public int IndexOf(double item)
        {
            return _data.IndexOf(item);
        }

        public void Insert(int index, double item)
        {
            _data.Insert(index, item);
        }

        public bool Remove(double item)
        {
            return _data.Remove(item);
        }

        public void RemoveAt(int index)
        {
            _data.RemoveAt(index);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void CopyData(Array array)
        {
            _data.Clear();
            _data.AddRange(array);
        }

        public double GetWeightSum()
        {
            double result = 0.0;
            foreach (var weight in _data)
            {
                result += weight;
            }

            return result;
        }

        #region Ariphmetic Operators

        private static void CountEqual(Array lh, Array rh)
        {
            if (rh.Count != lh.Count)
                throw new Exception("Size of Arrays are not equal");
        }

        public object Clone()
        {
            Array result = new Array();
            foreach (var i in this)
                result.Add(i);

            return result;
        }

        public IArray Subtract(double lh)
        {
            return lh - this;
        }

        public IArray Mul(IArray rh)
        {
            return this * (rh as Array);
        }

        public bool SetValue(int index, double value)
        {
            if (index >= Count)
                return false;

            _data[index] = value;
            return true;
        }

        public static Array operator *(Array lh, Array rh)
        {
            CountEqual(rh, lh);
            Array result = new Array(rh.Count, true);
            for (var i = 0; i < rh.Count; i++)
                result[i] = lh[i] * rh[i];
            return result;
        }

        public static Array operator *(Array lh, double rh)
        {
            Array result = new Array(lh.Count, true);

            for (var i = 0; i < lh.Count; i++)
                result[i] = lh[i] * rh;

            return result;
        }

        public static Array operator -(Array lh, Array rh)
        {
            CountEqual(lh, rh);
            Array result = new Array(lh.Count, true);
            for (int i = 0; i < lh.Count; i++)
                result[i] = lh[i] - rh[i];

            return result;
        }

        public static Array operator -(double lh, Array rh)
        {
            Array result = new Array(rh.Count, true);
            for (int i = 0; i < rh.Count; i++)
                result[i] = lh - rh[i];

            return result;
        }

        public static Array operator +(Array lh, Array rh)
        {
            Array result = new Array(lh.Count, true);
            for (var i = 0; i < lh.Count; i++)
                result[i] = lh[i] + rh[i];

            return result;
        }

        public static Array operator /(Array lh, double rh)
        {
            var result = new Array(lh.Count, true);
            for (var i = 0; i < lh.Count; i++)
                result[i] = lh[i] / rh;

            return result;
        }

        #endregion

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
