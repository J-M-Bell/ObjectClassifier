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
    var classifier: ClassifierModel
    ///class that selected as the prediction of the image
    var prediction = ""
    
    ///enumeration for possible errors
    enum DataError: Error {
        case missingData
    }
    
    init() {
        self.classifier = ClassifierModel(image: nil)
        classifyImage()
    }
    
    /// A helper function that classifies a UIImage based on the prediction from
    /// the Object Classifier Model
    private func classifyImage() {
        // TODO: Configure your classifier model as needed.
        // If your ClassifierModel has a failable or throwing initializer, update this accordingly.
        // Example placeholder initialization:
        let image = classifier.image
        var context: CGContext?
        var pointer: UnsafeMutablePointer<UInt8>?
        var bytesPerPixel: Int?
        if let image = image, let result = try? getImageInformation(image: image) {
            context = result.0
            pointer = result.1
            bytesPerPixel = result.2
        } else {
            // Handle missing image or extraction failure as needed
            // e.g., keep values nil or log a message
            print("Failed to extract image information")
        }
        var multiArray: MLMultiArray?
        multiArray = try? makeMultiArrayFromPixelData(context: context!, pixelDataPointer: pointer!, bytesPerPixel: bytesPerPixel!)
        findPrediction(multiArray: multiArray!)
    }
    
    
    /// A helper function that processes an image to be made into a MLMultiArray
    ///  - Parameter image: UIImage - image passed in from the user
    ///
    ///  - Returns the CGContext of the image
    ///  - Returns a pointer to the pixel data
    ///  - Returns the bytes per pixel
    ///
    ///  - Throws missing data error
    private func getImageInformation(image: UIImage) throws -> (CGContext, UnsafeMutablePointer<UInt8>, Int) {
        guard let cgImage = image.cgImage else {
            print("Could not convert image to CGImage")
            throw DataError.missingData
        }

        let width = Int(image.size.width)
        let height = Int(image.size.height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitsPerComponent = 8
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = bytesPerPixel * width
        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

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

        // Draw the image into the context (resize to 32x32 if needed by your model)
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: 32, height: 32))

        guard let data = context.data else {
            print("Could not create data pointer")
            throw DataError.missingData
        }
        let pixelData = data.bindMemory(to: UInt8.self, capacity: context.bytesPerRow * context.height)
        return (context, pixelData, bytesPerPixel)
    }
    
    /// A helper function that convert pixel data into a MLMultiArray
    /// - Parameters: context: CGContext - context object of CGImage
    /// - Parameters: pixelDataPointer: UnsafeMutablePointer<UInt8> - pointer to the memory location of the pixel data
    /// - Parameters: bytesPerPixel: Int - number of byte per pixel
    ///
    /// - Returns: MLMultiArray of RGB pixel value of the image passed in by the user
    ///
    /// - Throws: missing error if data is missing
    private func makeMultiArrayFromPixelData(context: CGContext, pixelDataPointer: UnsafeMutablePointer<UInt8>, bytesPerPixel: Int) throws -> MLMultiArray? {
        
        // 4. Calculate the offset for the specific pixel
        var pixelArray = [Float]()
        var x_r_Array = [Float]()
        var x_g_Array = [Float]()
        var x_b_Array = [Float]()
        
        var y_r_Array = [Float]()
        var y_g_Array = [Float]()
        var y_b_Array = [Float]()
        
        for y in 1...32 {
            for x in 1...32 {
                let pixelInfo: Int = ((context.bytesPerRow * y) + x * bytesPerPixel)
                let r = Float(pixelDataPointer[pixelInfo]) / 255.0
                x_r_Array.append(r)
                let g = Float(pixelDataPointer[pixelInfo + 1]) / 255.0
                x_g_Array.append(g)
                let b = Float(pixelDataPointer[pixelInfo + 2]) / 255.0
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
        
        // The 'shape' must match the total number of elements in the 'floatArray'.
        guard let shapedArray = try? MLShapedArray<Float>(scalars: pixelArray, shape: [1, 32, 32, 3])
        else {
            print("Error creating MLShapedArray. Check that the shape matches the array count.")
            throw DataError.missingData
        }
        // 2. Convert the MLShapedArray to an MLMultiArray using its initializer.
        let multiArray = MLMultiArray(shapedArray)
        return multiArray
    }
    
    /// A helper function that find the prediction from the object classifier model based on the values in the MLMultiArray
    /// - Parameters: multiArray: MLMultiArray - array of pixel data for model input
    private func findPrediction(multiArray: MLMultiArray) {
        do {
            var model = self.classifier.model
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
            self.prediction = self.classifier.classes[indexOfMax!]
        } catch DataError.missingData {
            print("Error: Missing input data.")
        } catch {
            print(error.localizedDescription)
        }
    }

}
