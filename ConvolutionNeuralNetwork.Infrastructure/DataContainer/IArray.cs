using System;
using System.Collections;

namespace ConvolutionNeuralNetwork.Infrastructure.DataContainer
{
    public interface IArray : ICloneable, IEnumerable
    {
        /// <summary>  Subtract IArray from double value (double - IArray)  </summary>
        IArray Subtract(double rh);
        IArray Mul(IArray array);
        bool SetValue(int index, double value);
        int Count { get; }
        double this[int i] { get; set; }
        string ToString();
    }
}
