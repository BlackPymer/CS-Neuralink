# NeuralNetworkCS

A comprehensive C# implementation of a neural network framework with support for arbitrary numeric types and flexible architecture design.

## Features

- **Generic Type Support**: Works with any numeric type (double, float, decimal, etc.)
- **Modular Architecture**: Clean separation of concerns with layers, operations, and loss functions
- **Flexible Layer System**: Support for various layer types including linear and sigmoid layers
- **Extensible Operations**: Easy to add new operations through inheritance or function delegates
- **Comprehensive Error Handling**: Detailed exception handling with meaningful error messages
- **Matrix Operations**: Efficient 2D matrix operations with generic type support

## Project Structure

```
NeuralNetwork/
├── NeuralNetwork/           # Core neural network implementation
│   ├── NeuralNetwork.cs     # Main neural network class
│   └── Exceptions.cs        # Custom exceptions
├── Layers/                  # Layer implementations
│   ├── Layer.cs            # Base layer class
│   ├── LinearLayer.cs      # Linear transformation layer
│   ├── SigmoidLayer.cs     # Sigmoid activation layer
│   └── Exceptions.cs       # Layer-specific exceptions
├── Operations/             # Operation implementations
│   ├── Operation.cs        # Base operation class
│   ├── ParamOperations/    # Parameterized operations
│   │   ├── ParamOperation.cs
│   │   ├── Weights.cs      # Weight matrix operation
│   │   ├── Bias.cs         # Bias addition operation
│   │   └── FuncParamOperation.cs
│   └── SimpleOperations/   # Stateless operations
│       ├── SimpleOperation.cs
│       ├── SigmoidOperation.cs
│       └── FuncSimpleOperation.cs
├── Losses/                 # Loss function implementations
│   ├── Loss.cs            # Base loss class
│   └── MeanSquaresError.cs # MSE loss implementation
├── Matrix.cs              # 2D matrix implementation
└── Program.cs             # Example usage
```

## Core Components

### NeuralNetwork<T>
The main class that orchestrates the neural network training and inference process.

**Key Methods:**
- `Train()`: Trains the network using gradient descent
- `Test()`: Evaluates network performance on test data
- `GetOutput()`: Performs inference on a single input

### Layer<T>
Base class for all neural network layers. Supports forward and backward propagation.

**Available Layer Types:**
- `LinearLayer<T>`: Linear transformation with weights and bias
- `SigmoidLayer<T>`: Sigmoid activation with optional linear transformation

### Operations
Operations define the mathematical transformations applied within layers.

**Parameterized Operations:**
- `Weights<T>`: Matrix multiplication with learnable weights
- `Bias<T>`: Element-wise bias addition

**Simple Operations:**
- `SigmoidOperation<T>`: Sigmoid activation function

### Loss Functions
Implementations of loss functions for measuring prediction accuracy.

**Available Loss Functions:**
- `MSE<T>`: Mean Squared Error loss

## Usage Example

```csharp
using NeuralNetwork.Layers;
using NeuralNetwork.Losses;

// Create a neural network with sigmoid layers
var neuralNetwork = new NeuralNetwork<double>(new List<Layer<double>>
{
    new SigmoidLayer<double>(3, 64),    // Input: 3, Hidden: 64
    new SigmoidLayer<double>(64, 4)     // Hidden: 64, Output: 4
}, new MSE<double>());

// Prepare training data
var trainInputs = new List<Matrix2d<double>>();
var trainOutputs = new List<Matrix2d<double>>();
// ... populate with your data

// Train the network
neuralNetwork.Train(trainInputs, trainOutputs, 
    epoches: 1000, 
    applyEvery: 100, 
    learningRate: 0.01);

// Test the network
var testInputs = new List<Matrix2d<double>>();
var testOutputs = new List<Matrix2d<double>>();
// ... populate with test data

double testLoss = neuralNetwork.Test(testInputs, testOutputs);
Console.WriteLine($"Test Loss: {testLoss}");

// Make predictions
var input = new Matrix2d<double>(1, 3); // Single sample
var prediction = neuralNetwork.GetOutput(input);
```

## Matrix Operations

The `Matrix2d<T>` class provides comprehensive matrix operations:

- **Basic Operations**: Addition, subtraction, multiplication, transpose
- **Element-wise Operations**: Custom function application
- **Initialization**: Random values, zeros, ones, custom values
- **Utility Methods**: Copy, sum, element enumeration

## Error Handling

The framework provides comprehensive error handling:

- **BatchSizeMismatchError**: When input/output batch sizes don't match
- **LayerWrongInputSize**: When layer input dimensions are incorrect
- **LayerOperationsMismatch**: When operation dimensions are incompatible
- **ArgumentNullException**: When required parameters are null
- **ArgumentException**: When parameters have invalid values

## Building and Running

1. **Prerequisites**: .NET Framework or .NET Core
2. **Build**: Use Visual Studio or `dotnet build`
3. **Run**: Execute the compiled executable or use `dotnet run`

## Example Application

The included `Program.cs` demonstrates RGB to CMYK color conversion using a neural network:

- Generates random RGB values (0-1 range)
- Converts them to CMYK using mathematical formulas
- Trains a neural network to learn this conversion
- Tests the network's accuracy

## Extending the Framework

### Adding New Layers
```csharp
public class CustomLayer<T> : Layer<T>
{
    public CustomLayer(int inputSize, int outputSize) 
        : base(inputSize, new List<Operation<T>> { /* your operations */ })
    {
        LayerSize = outputSize;
    }
}
```

### Adding New Operations
```csharp
public class CustomOperation<T> : SimpleOperation<T>
{
    protected override Matrix2d<T> CalculateOutput(Matrix2d<T> input)
    {
        // Your forward pass implementation
    }

    protected override Matrix2d<T> CalculateDerivative(Matrix2d<T> dOutput)
    {
        // Your backward pass implementation
    }
}
```

### Adding New Loss Functions
```csharp
public class CustomLoss<T> : Loss<T>
{
    public override dynamic CalculateLoss(Matrix2d<T> prediction, Matrix2d<T> correctResult)
    {
        // Your loss calculation
    }

    public override Matrix2d<T> CalculateLossGradient(Matrix2d<T> prediction, Matrix2d<T> correctResult)
    {
        // Your gradient calculation
    }
}
```

## Performance Considerations

- Uses dynamic arithmetic for generic types - consider specialized implementations for production use
- Matrix operations are not optimized for large-scale applications
- Consider using specialized libraries like ML.NET for production neural network applications

## License

This project is provided as-is for educational and research purposes.