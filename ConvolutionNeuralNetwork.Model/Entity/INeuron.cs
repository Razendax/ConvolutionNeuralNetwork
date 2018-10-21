using ConvolutionNeuralNetwork.Infrastructure.DataContainer;

namespace ConvolutionNeuralNetwork.Model.Entity
{
    public abstract class INeuron
    {
        public abstract double FormOutput(Array3D array);

        public double Bias { get; set; }
        public Array3D Weights { get; set; }
    }
}
