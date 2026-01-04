//
//  ClassifierViewModel.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/29/25.
//

import Foundation
import SwiftUI
import CoreML

/// A view model that handles the data functions of the classifier view and model
class ClassifierViewModel {
    
    /// classifier that is passed in from view
    private var classifier: ClassifierModel
    /// Object Classifier model object
    private let model: ObjectClassifierModel = {
        do {
            let config = MLModelConfiguration()
            return try ObjectClassifierModel(configuration: config)
        } catch {
            print(error)
            fatalError("Error loading model")
        }
    }()
    ///enumeration for possible errors
    enum DataError: Error {
        case missingData
    }
    
    
    init() {
        self.classifier = ClassifierModel()
    }
    
    /// A function that process a image and returns the class that was
    /// predicted by the Object Classifier Model
    func classifyImage(image: UIImage) -> String? {
        let multiArray = try? processImage(image: image)
        let prediction = try? getPrediction(multiArray: multiArray!)
        return prediction
    }
    
    /// A helper function that processes an image to be made into a MLMultiArray
    ///  - Parameter image: UIImage - image passed in from the user
    ///
    /// - Returns: MLMultiArray of RGB pixel value of the image passed in by the user
    ///
    /// - Throws: missing data error
    private func processImage(image: UIImage) throws -> MLMultiArray? {
        // convert UIImage into CGImage
        guard let cgImage = image.cgImage else {
            print("Could not convert image to CGImage")
            throw DataError.missingData
        }

        // create dimension variables for CGContext
        let width = Int(image.size.width)
        let height = Int(image.size.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitsPerComponent = 8
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = bytesPerPixel * width
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

        // Create CGContext for image processing
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else {
            print("Could not create bitmap context")
            throw DataError.missingData
        }

        // Draw the image into the context to resize the image to 32x32
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: 32, height: 32))

        // Create data pointer to bind to pixel data memory location
        guard let data = context.data else {
            print("Could not create data pointer")
            throw DataError.missingData
        }
        let pixelData = data.bindMemory(to: UInt8.self, capacity: context.bytesPerRow * context.height)
        
        
        /// Create multiple arrays to hold different pixel values depending
        /// on the RBG channels of the image to be used to create the
        /// MLMultiArray input to the model
        var pixelArray = [Float]()
        var x_r_Array = [Float]()
        var x_g_Array = [Float]()
        var x_b_Array = [Float]()
        
        var y_r_Array = [Float]()
        var y_g_Array = [Float]()
        var y_b_Array = [Float]()
        
        /// Loop to cycle through the pixel data and store that data in the arrays
        /// then append them together
        for y in 1...32 {
            for x in 1...32 {
                let pixelInfo: Int = ((context.bytesPerRow * y) + x * bytesPerPixel)
                let r = Float(pixelData[pixelInfo]) / 255.0
                x_r_Array.append(r)
                let g = Float(pixelData[pixelInfo + 1]) / 255.0
                x_g_Array.append(g)
                let b = Float(pixelData[pixelInfo + 2]) / 255.0
                x_b_Array.append(b)
            }
            y_r_Array.append(contentsOf: x_r_Array)
            y_g_Array.append(contentsOf: x_g_Array)
            y_b_Array.append(contentsOf: x_b_Array)
            x_r_Array.removeAll()
            x_g_Array.removeAll()
            x_b_Array.removeAll()
        }
        pixelArray.append(contentsOf: y_r_Array)
        pixelArray.append(contentsOf: y_g_Array)
        pixelArray.append(contentsOf: y_b_Array)
        
        // Create MLShapedArray from pixel data array
        guard let shapedArray = try? MLShapedArray<Float>(scalars: pixelArray, shape: [1, 32, 32, 3])
        else {
            print("Error creating MLShapedArray. Check that the shape matches the array count.")
            throw DataError.missingData
        }
        // Convert the MLShapedArray to an MLMultiArray
        let multiArray = MLMultiArray(shapedArray)
        
        return multiArray
    }

    
    /// A helper function that find the prediction from the object classifier model based on the values in the MLMultiArray
    /// - Parameters: multiArray: MLMultiArray - array of pixel data for model input
    ///
    /// - Returns: the prediction of the ObjectClassifierModel object
    private func getPrediction(multiArray: MLMultiArray) -> String {
        var prediction = "Could not get prediction."
        do {
            let model = self.model
            let input = ObjectClassifierModelInput(keras_tensor_50: multiArray)
            let output = try model.prediction(input: input)
            let outputArray = output.Identity
            var probs = [Float]()
            for i in 0..<outputArray.count {
                let value = outputArray[i].floatValue
                probs.append(value)
            }
            
            let maxElement = probs.max()
            let indexOfMax = probs.firstIndex(of: maxElement!)
            prediction = self.classifier.classes[indexOfMax!]
        } catch DataError.missingData {
            print("Error: Missing input data.")
        } catch {
            print(error.localizedDescription)
        }
        return prediction
    }

}
