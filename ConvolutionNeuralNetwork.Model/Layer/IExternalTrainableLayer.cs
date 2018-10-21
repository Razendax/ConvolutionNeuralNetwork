using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Model.Training;

namespace ConvolutionNeuralNetwork.Model.Layer
{
    public interface IExternalTrainableLayer : ITrainableLayer<Array3D, Array3D>
    {
        Trainer Trainer { get; set; }
    }
}
