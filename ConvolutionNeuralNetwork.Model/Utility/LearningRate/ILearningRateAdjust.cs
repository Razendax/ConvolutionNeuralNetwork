
namespace ConvolutionNeuralNetwork.Model.Utility.LearningRate
{
    public interface ILearningRateAdjust
    {
        double Adjust(int iteration);
    }
}
