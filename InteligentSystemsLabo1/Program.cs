using System;
using System.Diagnostics.Metrics;

class Perceptron
{
    private double w1, w2, b; // Weights and bias
    private double learningRate; // Learning rate (eta)

    // Constructor to initialize perceptron with random weights and learning rate
    public Perceptron(double learningRate)
    {
        Random random = new Random();
        w1 = random.NextDouble() * 2 - 1; // Initialize between -1 and 1
        w2 = random.NextDouble() * 2 - 1;
        b = random.NextDouble() * 2 - 1;
        this.learningRate = learningRate;
    }

    // Activation function based on the equation provided
    public int Predict(double x1, double x2)
    {
        double sum = x1 * w1 + x2 * w2 + b;
        return sum > 0 ? 1 : -1;
    }

    // Training algorithm
    public void Train(double[,] inputs, int[] labels, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            int totalError = 0;

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                double x1 = inputs[i, 0];
                double x2 = inputs[i, 1];
                int desiredOutput = labels[i];

                // Predict current output
                int predictedOutput = Predict(x1, x2);

                // Calculate error
                int error = desiredOutput - predictedOutput;

                // Update weights and bias if there's an error
                if (error != 0)
                {
                    totalError += Math.Abs(error);
                    w1 += learningRate * error * x1;
                    w2 += learningRate * error * x2;
                    b += learningRate * error;
                }
            }

            // Optional: Display progress
            if (totalError == 0)
            {
                Console.WriteLine($"Training complete at epoch {epoch} with no errors.");
                break;
            }
        }
    }

    static void Main(string[] args)
    {
        // Training data (2 features per object)
        double[,] inputs = {
            { 0.1, 0.5 },
            { 0.3, 0.7 },
            { 0.6, 0.9 },
            { 0.4, 0.2 },
            { 0.9, 0.6 },
            { 0.7, 0.1 }
        };

        // Desired outputs (labels)
        int[] labels = { 1, 1, 1, -1, -1, -1 };

        // Create the perceptron with a learning rate of 0.1
        Perceptron perceptron = new Perceptron(0.1);

        // Train the perceptron
        perceptron.Train(inputs, labels, epochs: 100);

        // Test the trained perceptron
        Console.WriteLine("Testing the trained Perceptron:");
        for (int i = 0; i < inputs.GetLength(0); i++)
        {
            double x1 = inputs[i, 0];
            double x2 = inputs[i, 1];
            int output = perceptron.Predict(x1, x2);
            Console.WriteLine($"Input: ({x1}, {x2}), Predicted Output: {output}, Expected: {labels[i]}");
        }
    }
}

