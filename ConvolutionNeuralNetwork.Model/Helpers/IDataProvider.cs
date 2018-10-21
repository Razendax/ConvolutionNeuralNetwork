
using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Helpers
{
    public interface IDataProvider<TIn, TOut>
        where TIn : IArray
        where TOut : IArray
    {
        TrainingData<TIn, TOut> GetNext(bool isTraining);
    }
}
