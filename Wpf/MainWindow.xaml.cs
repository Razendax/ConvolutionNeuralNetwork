using System;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;

using ConvolutionNeuralNetwork.Infrastructure.DataContainer;
using ConvolutionNeuralNetwork.Infrastructure.Functions;
using ConvolutionNeuralNetwork.Model.Helpers;
using ConvolutionNeuralNetwork.Model.Layer;
using ConvolutionNeuralNetwork.Model.Network;
using ConvolutionNeuralNetwork.Model.Training;
using Wpf.Annotations;

namespace Wpf
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        private const int BatchSize = 4;
        private const int EpochSize = 16;
        public MainWindow()
        {
            InitializeComponent();
            InitializeNetwork();
        }

        private void InitializeNetwork()
        {
            var inputLayer = new InputLayer3D(1, 1, 1);
            var outputLayer = new OutputLayer(1) { ActivationFunction = new ConstOutputArrayFunction() };
            var dataProvider = new FunctionProvider
            {
                TrainData = {
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(0, 0), Expected = new Array3D(0.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(0, 1), Expected = new Array3D(1.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(1, 0), Expected = new Array3D(1.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D (1, 1), Expected = new Array3D(0.0)
                }
            },
                TestData = { new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D(0, 0), Expected = new Array3D(0)
                    },
                    new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D(0, 1), Expected = new Array3D(1)
                    },
                    new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D(1, 0), Expected = new Array3D(1)
                    },
                    new TrainingData<Array3D, Array3D>
                    {
                        Input = new Array3D (1, 1), Expected = new Array3D(0)
                    } },
                IsQueue = false
            };
            var oneData = new FunctionProvider
            {
                TrainData = {
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(0, 0), Expected = new Array3D(0.0)
                },
                new TrainingData<Array3D, Array3D>
                {
                    Input = new Array3D(1,1), Expected = new Array3D(0.0)
                } },
                IsQueue = false
            };
            var function = new FunctionProvider(x => Math.Pow(x, 2));

            var perceptron1 = new PerceptronLayer(10, 1) { ActivationFunction = new TanhActivationFunction() };
            perceptron1.Trainer = new MiniBatchPerceptronTrainer(perceptron1.Neurals, false)
            { BatchSize = BatchSize, ActivationFunction = new TanhActivationFunction(), LearningRate = 0.1, Momentum = 0.1 };
            var perceptron2 = new PerceptronLayer(1, 10) { ActivationFunction = new TanhActivationFunction() };
            perceptron2.Trainer = new MiniBatchPerceptronTrainer(perceptron2.Neurals, true)
            { BatchSize = BatchSize, ActivationFunction = new TanhActivationFunction(), LearningRate = 0.1, Momentum = 0.1 };

            Network = new MultiLayerPerceptron
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer,
                DataProvider = function
            };
            Network.HiddenLayers.Add(perceptron1);
            Network.HiddenLayers.Add(perceptron2);

            Trainer = new FCTrainer(Network, EpochSize, BatchSize, function);
        }

        private MultiLayerPerceptron _network;
        public MultiLayerPerceptron Network
        {
            get => _network;
            set
            {
                _network = value;
                OnPropertyChanged(nameof(Network));
            }
        }

        private FCTrainer _trainer;
        public FCTrainer Trainer
        {
            get => _trainer;
            set
            {
                _trainer = value;
                OnPropertyChanged(nameof(Trainer));
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void ButtonBase_OnClick(object sender, RoutedEventArgs e)
        {
            
        }
    }
}
