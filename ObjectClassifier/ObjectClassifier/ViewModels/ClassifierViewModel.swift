//
//  ClassifierViewModel.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/26/25.
//

import Foundation
import TensorFlowLite

class TFLiteModelLoader {
    private var interpreter: Interpreter?

    init?(modelName: String) {
        // Find the model path in the app bundle
        guard let modelPath = Bundle.main.path(forResource: modelName, ofType: "tflite") else {
            print("Failed to find the model file in the bundle.")
            return nil
        }

        do {
            // Initialize the interpreter with the model path
            interpreter = try Interpreter(modelPath: modelPath)
            print("TensorFlow Lite model loaded successfully.")
            
            // Optional: Allocate tensors
            try interpreter?.allocateTensors()
        } catch let error {
            print("Failed to load the model or allocate tensors: \(error.localizedDescription)")
            return nil
        }
    }

    // You can add methods here to run inference, process input, and interpret output
    
    // Example of running inference (simplified)
    func runInference(inputData: Data) -> Data? {
        do {
            try interpreter?.copy(inputData, toInputAt: 0) // Copy data to input tensor at index 0
            try interpreter?.invoke() // Run the model
            let outputTensor = try interpreter?.output(at: 0) // Get the output tensor at index 0
            return outputTensor?.data
        } catch {
            print("Inference failed: \(error.localizedDescription)")
            return nil
        }
    }

}
