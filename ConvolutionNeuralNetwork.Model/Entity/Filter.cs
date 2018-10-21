using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Entity
{
    public class Filter : INeuron
    {
        public Filter() { }

        public Filter(int height, int width, int depth)
        {
            Weights = new Array3D(height, width, depth);
        }

        public override double FormOutput(Array3D array)
        {
            throw new System.NotImplementedException();
        }
    }
}
