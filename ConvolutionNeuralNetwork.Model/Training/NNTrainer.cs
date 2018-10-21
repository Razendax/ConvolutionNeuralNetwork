using System.Collections.Generic;
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Helpers;

namespace ConvolutionNeuralNetwork.Model.Training
{
    public abstract class NNTrainer<TIn, TOut>
        where TIn : IArray
        where TOut : IArray
    {
        protected int EpochSize { get; set; }
        protected int BatchSize { get; set; }
        public IDataProvider<TIn, TOut> DataProvider { get; }

        protected NNTrainer(int epoch, int batch, IDataProvider<TIn, TOut> dataProvider)
        {
            EpochSize = epoch;
            BatchSize = batch;
            DataProvider = dataProvider;
        }

        protected abstract TOut ForwardPropagation(TIn input, out List<TOut> outputs);
        protected abstract void BackPropagation(TIn input, TOut expected, TOut actual, List<TOut> outputs);
        public abstract void Train(int epochs);
    }
}
