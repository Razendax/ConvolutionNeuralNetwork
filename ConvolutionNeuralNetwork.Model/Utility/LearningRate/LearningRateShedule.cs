
namespace ConvolutionNeuralNetwork.Model.Utility.LearningRate
{
    public abstract class LearningRateShedule : ILearningRateAdjust
    {
        public double Max { get; set; }
        public double Min { get; set; }
        public double Step { get; set; }
        public abstract double Adjust(int iteration);
    }
}
