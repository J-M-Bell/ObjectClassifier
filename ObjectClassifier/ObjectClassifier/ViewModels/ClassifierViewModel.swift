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
    
    /// ClassifierViewModel class's main function that process a image and returns the class that was
    /// predicted by the Object Classifier Model
    /// - Parameters: image: UIImage - UIImage from user
    ///
    /// - Returns: the prediction from the ObjectClassifierModel object
    func classifyImage(image: UIImage) -> String? {
        do {
            let processedImageWidth = 32
            let processedImageHeight = 32
            guard let multiArray = try processImage(image: image, processedImageWidth: processedImageWidth, processedImageHeight: processedImageHeight) else { return nil }
            let probs = try getModelOutput(multiArray: multiArray)
            let prediction = try getPrediction(probs: probs)
            return prediction
        } catch {
            print("Error processing image: \(error)")
            return nil
        }
    }
    
    /// A helper function that processes an image to be made into a MLMultiArray
    /// Note: do not call this function, use classifyImage method
    ///  - Parameter image: UIImage - image passed in from the user
    ///
    /// - Returns: MLMultiArray of RGB pixel value of the image passed in by the user
    ///
    /// - Throws: missing data error
    func processImage(image: UIImage, processedImageWidth: Int, processedImageHeight: Int) throws -> MLMultiArray? {
            
        // convert UIImage into CGImage
        guard let cgImage = image.cgImage else {
            print("Could not convert image to CGImage")
            throw DataError.missingData
        }

        // resize CGImage to correct size for model input
        let newSize = CGSize(width: processedImageWidth, height: processedImageHeight)
        let resizedCGImage = resizeCGImage(image: cgImage, newSize: newSize)
        
        // create dimension variables for CGContext
        let width = processedImageWidth
        let height = processedImageHeight
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitsPerComponent = 8
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = bytesPerPixel * width
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        let imgPointer = UnsafeMutableRawPointer.allocate(
                byteCount: bytesPerRow * height,
                alignment: MemoryLayout<UInt8>.alignment)
        
        
        // Create CGContext for image processing
        guard let context = CGContext(
            data: imgPointer,
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

        // Draw the image into the context
        context.draw(resizedCGImage!, in: CGRect(x: 0, y: 0, width: processedImageWidth, height: processedImageHeight))

        // Create data pointer from CGContext
        guard let data = context.data else {
            print("Could not create data pointer")
            throw DataError.missingData
        }
        
        // Creating a bound data pointer to memory location where pixel data is stored
        let pixelData = data.bindMemory(to: UInt8.self, capacity: context.bytesPerRow * context.height)
        
        
        // Main array to hold final RGB pixel values in the correct order
        var pixelArray = [Float]()
        
        // arrays that hold the rgb values per row of pixel data
        var x_r_Array = [Float]()
        var x_g_Array = [Float]()
        var x_b_Array = [Float]()
        
        // arrays that will contain the entire pixel data for each channel
        var y_r_Array = [Float]()
        var y_g_Array = [Float]()
        var y_b_Array = [Float]()

        
        // Loop to cycle through the pixel data and store that data in the arrays then append them together
        for y in 0..<processedImageHeight // cycle through each row in the image
        {
            for x in 0..<processedImageWidth // cycle through each pixel in a row
            {
                // current pixel position
                let pixelInfo: Int = ((context.bytesPerRow * y) + x * bytesPerPixel)
                let r = Float(pixelData[pixelInfo]) / 255.0
                x_r_Array.append(r)
                let g = Float(pixelData[pixelInfo + 1]) / 255.0
                x_g_Array.append(g)
                let b = Float(pixelData[pixelInfo + 2]) / 255.0
                x_b_Array.append(b)
            }
            // Append contents from each row to y arrays
            y_r_Array.append(contentsOf: x_r_Array)
            y_g_Array.append(contentsOf: x_g_Array)
            y_b_Array.append(contentsOf: x_b_Array)
            // Remove values from x arrays to start over
            x_r_Array.removeAll()
            x_g_Array.removeAll()
            x_b_Array.removeAll()
        }
        // Append all three channels to final array
        pixelArray.append(contentsOf: y_r_Array)
        pixelArray.append(contentsOf: y_g_Array)
        pixelArray.append(contentsOf: y_b_Array)
        
        // Validate pixel count matches expected shape (1 x 32 x 32 x 3)
        let expectedCount = 1 * 32 * 32 * 3
        guard pixelArray.count == expectedCount else {
            print("Error: pixel array count (\(pixelArray.count)) does not match expected count (\(expectedCount)).")
            throw DataError.missingData
        }

        // Create MLShapedArray from pixel data array
        let shapedArray = MLShapedArray<Float>(scalars: pixelArray, shape: [1, 32, 32, 3])
        // Convert the MLShapedArray to an MLMultiArray
        let multiArray = MLMultiArray(shapedArray)

        return multiArray
    }

    
    /// A helper function that find the prediction from the object classifier model based on the values in the MLMultiArray
    /// Note: do not call this function, use classifyImage method
    /// - Parameters: multiArray: MLMultiArray - array of pixel data for model input
    ///
    /// - Returns: the prediction of the ObjectClassifierModel object as a String
    func getModelOutput(multiArray: MLMultiArray) throws -> [Float] {
        do {
            // get prediction from ObjectClassifierModel object
            let input = ObjectClassifierModelInput(keras_tensor_50: multiArray)
            let output = try model.prediction(input: input)
            let outputArray = output.Identity
            
            // Iterate through values from output and append to probabilities array
            var probs = [Float]()
            for i in 0..<outputArray.count {
                probs.append(outputArray[i].floatValue)
            }
            return probs
        } catch {
            print("Probs array not populated")
            print(error.localizedDescription)
            throw DataError.missingData
        }
    }
    
    /// Helper function to get prediction based on the max value from
    /// the probablity array output from the ObjectClassifierModel object
    /// - Parameters: probs: [Float] - array of probabilities from the ObjectClassifierModel object
    ///
    /// - Returns: the prediction as a String
    func getPrediction(probs: [Float]) throws -> String {
         //Find the index of the max value and set prediction to class that cooresponds to that index
        if let maxElement = probs.max(), let indexOfMax = probs.firstIndex(of: maxElement) {
            let prediction = self.classifier.classes[indexOfMax]
            return prediction
        } else {
            print("Could not find prediction" )
            throw DataError.missingData
        }
    }

    /// Helper function for resize CGImage so that the user input image and CGContext will
    /// match dimension before allocating the memory to be read by the data pointer
    /// - Parameters: image: CGImage - CGImage to be resized
    /// - Parameters: newSize: CGSize - new size of the CGImage
    ///
    /// - Returns: resized CGImage
    func resizeCGImage(image: CGImage, newSize: CGSize) -> CGImage? {
        // 1. Create a UIGraphicsImageRenderer for easily creating a new bitmap context.
        // The 'format' handles correct scaling for different device resolutions (use scale 1.0 for actual pixels).
        let renderer = UIGraphicsImageRenderer(size: newSize, format: UIGraphicsImageRendererFormat.default())
        
        let resizedImage = renderer.image { (context) in
            // 2. Get the Core Graphics context
            let cgContext = context.cgContext
            
            // 3. Optional: Set interpolation quality (high quality is default but can be explicit)
            cgContext.interpolationQuality = .high
            
            // 4. Draw the original CGImage into the new context, filling the entire new size
            // The image will be scaled to fit the destination rectangle
            let newRect = CGRect(origin: .zero, size: newSize)
            cgContext.draw(image, in: newRect)
        }
        
        // 5. Return the CGImage from the newly created UIImage
        return resizedImage.cgImage
    }
}
