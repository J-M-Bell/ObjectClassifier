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
        do {
            let processedImageWidth = 32
            let processedImageHeight = 32
            guard let multiArray = try processImage(image: image, processedImageWidth: processedImageWidth, processedImageHeight: processedImageHeight) else { return nil }
            let prediction = getPrediction(multiArray: multiArray)
            return prediction
        } catch {
            print("Error processing image: \(error)")
            return nil
        }
    }
    
    /// A helper function that processes an image to be made into a MLMultiArray
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

        let newSize = CGSize(width: processedImageWidth, height: processedImageHeight)
        let resizedCGImage = resizeCGImage(image: cgImage, newSize: newSize)
        // create dimension variables for CGContext
        let width = processedImageWidth
        let height = processedImageHeight
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitsPerComponent = 8
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = bytesPerPixel * width
//        let bitmapInfo = CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
//        let testBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | kCGBitmapByteOrder32Host.rawValue)
        let testBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
//        let bitmapInfo = CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Little.rawValue

        let imgPointer = UnsafeMutableRawPointer.allocate(
                byteCount: bytesPerRow * height,
                alignment: MemoryLayout<UInt8>.alignment)
        
//        var intArray = [UInt8](repeating: 0, count: dataSize)
        
        // Create CGContext for image processing
        guard let context = CGContext(
            data: imgPointer,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: testBitmapInfo
        ) else {
            print("Could not create bitmap context")
            throw DataError.missingData
        }

        // Draw the image into the context to resize the image to 32x32
        context.draw(resizedCGImage!, in: CGRect(x: 0, y: 0, width: processedImageWidth, height: processedImageHeight))

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
//        var x_a_Array = [Float]()
        
        var y_r_Array = [Float]()
        var y_g_Array = [Float]()
        var y_b_Array = [Float]()
//        var y_a_Array = [Float]()
        
        /// Loop to cycle through the pixel data and store that data in the arrays
        /// then append them together
        for y in 0..<processedImageHeight {
            for x in 0..<processedImageWidth {
                let pixelInfo: Int = ((context.bytesPerRow * y) + x * bytesPerPixel)
                var r = Float(pixelData[pixelInfo])
                r /= 255.0
                x_r_Array.append(r)
                var g = Float(pixelData[pixelInfo + 1])
                g /= 255.0
                x_g_Array.append(g)
                var b = Float(pixelData[pixelInfo + 2])
                b /= 255.0
                x_b_Array.append(b)
//                let a = Float(pixelData[pixelInfo + 3]) / 255.0
//                x_a_Array.append(a)
                
            }
            y_r_Array.append(contentsOf: x_r_Array)
            y_g_Array.append(contentsOf: x_g_Array)
            y_b_Array.append(contentsOf: x_b_Array)
//            y_a_Array.append(contentsOf: x_a_Array)
            x_r_Array.removeAll()
            x_g_Array.removeAll()
            x_b_Array.removeAll()
//            x_a_Array.removeAll()
        }
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
    /// - Parameters: multiArray: MLMultiArray - array of pixel data for model input
    ///
    /// - Returns: the prediction of the ObjectClassifierModel object
    func getPrediction(multiArray: MLMultiArray) -> String {
        var prediction = "Could not get prediction."
        do {
            let input = ObjectClassifierModelInput(keras_tensor_50: multiArray)
            let output = try model.prediction(input: input)
            let outputArray = output.Identity
            var probs = [Float]()
            for i in 0..<outputArray.count {
                probs.append(outputArray[i].floatValue)
            }
            if let maxElement = probs.max(), let indexOfMax = probs.firstIndex(of: maxElement) {
                prediction = self.classifier.classes[indexOfMax]
            }
        } catch {
            print(error.localizedDescription)
        }
        return prediction
    }

    /// Helper function for resize CGImage so that the user input image and CGContext will
    /// match dimension before allocating the memory to be read by the data pointer
    /// - Parameters: image: CGImage - CGImage to be resized
    /// - Parameters: newSize: CGSize - new size of the CGImage
    ///
    /// - Returns: resized CGImage
    private func resizeCGImage(image: CGImage, newSize: CGSize) -> CGImage? {
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
