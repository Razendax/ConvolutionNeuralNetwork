
namespace ConvolutionNeuralNetwork.Model.Utility.LearningRate
{
    public class StepDecay : LearningRateShedule
    {
        public override double Adjust(int iteration)
        {
            return Max - (Max - Min) / Step * (iteration % Step);
        }
    }
}
