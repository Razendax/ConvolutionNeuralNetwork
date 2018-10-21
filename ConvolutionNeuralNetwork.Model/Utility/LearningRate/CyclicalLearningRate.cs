
using System;

namespace ConvolutionNeuralNetwork.Model.Utility.LearningRate
{
    public class CyclicalLearningRate : LearningRateShedule
    {
        public double Decay { get; set; }

        public override double Adjust(int iteration)
        {
            var firstHalf = iteration <= Step / 2;
            var result = 0.0;
            if (firstHalf)
                result = Min + (Max - Min) / Step * 2 * iteration;
            else
                result = Max - (Max - Min) / Step * 2 * (iteration - Step / 2) ;

            if (Math.Abs(iteration - Step) < 0.1)
                Max -= Decay;
            return result;
        }
    }
}
