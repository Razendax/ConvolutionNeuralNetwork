using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;

namespace ConvolutionNeuralNetwork.Model.Network
{
    public abstract class INN <TIn, TOut>
        where TIn : class, IArray
        where TOut : class, IArray
    {
        public IInputLayer<TIn, TIn> InputLayer { get; set; }
        public IOutputLayer<TOut, TOut> OutputLayer { get; set; }
        public IDataProvider<TIn, TOut> DataProvider { get; set; }

        public List<ITrainableLayer<TIn, TOut>> HiddenLayers { get; } = new List<ITrainableLayer<TIn, TOut>>();

        public virtual TOut ForwardPropagation(TIn input)
        {
            IArray linput = InputLayer.FormOutput(input);
            foreach (var layer in HiddenLayers)
            {
                linput = layer.FormOutput(linput);
            }

            return OutputLayer.FormOutput(linput as TOut);
        }

        public virtual double Test(int iterations = 1)
        {
            var error = 0.0;
            for (var i = 0; i < iterations; i++)
            {
                var pair = DataProvider.GetNext(false);
                var actual = ForwardPropagation(pair.Input);
                error += OutputLayer.GetNetworkError(pair.Expected, actual);
            }

            return error / iterations;
        }
    }
}
