//
//  ObjectClassifierTests.swift
//  ObjectClassifierTests
//
//  Created by Justin Bell on 12/28/25.
//

import Testing
import SwiftUI
import CoreML
import Foundation
@testable import ObjectClassifier

class ClassifierViewModelTests {

    
    @Test func example() async throws {
        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
    }
    
    @Test(.tags(.unitTests)) func processImageTest() throws {
        let vm = ClassifierViewModel()
        let width = 40
        let height = 40
        let processedImageWidth = 32
        let processedImageHeight = 32
        let image = UIImage.from(color: UIColor.white, width: width, height: height)
//        let image2 = UIImage.from(color: UIColor.white, size: [32, 32])
        let testMultiArray = try fillMLMultiArray(shape: [1, 32, 32, 3], withValue: 1)
        let actualMultiArray = try vm.processImage(image: image!, processedImageWidth: processedImageWidth, processedImageHeight: processedImageHeight)
        let areEqual = areMLMultiArraysEqual(testMultiArray, actualMultiArray!)
        #expect(areEqual)
//        #expect(testMultiArray.count == actualMultiArray?.count)
    }


    /// Helper testing function that takes in two MLMultiArray objects and determines
    /// if those object have all of the same values
    /// - Parameters: multiArray1: MLMultiArray - the first MLMultiArray object
    /// - Parameters: multiArray2: MLMultiArray -  the second MLMultiArray object
    ///
    /// - Returns: true or false depending on if the values are the same
    func areMLMultiArraysEqual(
        _ multiArray1: MLMultiArray,
        _ multiArray2: MLMultiArray
    ) -> Bool {
        // 1. Check if shapes and data types match
        guard multiArray1.shape == multiArray2.shape else {
            return false
        }

        guard multiArray1.dataType == multiArray2.dataType else {
            return false
        }

        // 2. Compare elements based on data type
        let count = multiArray1.count
        
        switch multiArray1.dataType {
        case .float32:
            let ptr1 = UnsafeMutablePointer<Float>(OpaquePointer(multiArray1.dataPointer))
            let ptr2 = UnsafeMutablePointer<Float>(OpaquePointer(multiArray2.dataPointer))
            for i in 0..<count {
                if ptr1[i] != ptr2[i] { return false }
            }
        case .double:
            let ptr1 = UnsafeMutablePointer<Double>(OpaquePointer(multiArray1.dataPointer))
            let ptr2 = UnsafeMutablePointer<Double>(OpaquePointer(multiArray2.dataPointer))
            for i in 0..<count {
                if ptr1[i] != ptr2[i] { return false }
            }
        case .int32:
            let ptr1 = UnsafeMutablePointer<Int32>(OpaquePointer(multiArray1.dataPointer))
            let ptr2 = UnsafeMutablePointer<Int32>(OpaquePointer(multiArray2.dataPointer))
            for i in 0..<count {
                if ptr1[i] != ptr2[i] { return false }
            }
        // Add other cases like .float16 if needed, though they require specific handling
        default:
            // Handle unsupported or unknown types
            print("Unsupported MLMultiArray data type for comparison.")
            return false
        }
        
        return true
    }
    
    /// Helper function to run for testing ClassiferViewModel processImage method
    /// - Parameters: shape: [Int} - array of ints declaring the shape of the multiArray
    ///
    /// - Returns: MLMultiArray object that is filled with zeros
    func fillMLMultiArray(shape: [Int], withValue: Int) throws -> MLMultiArray {
        let multiArray = try MLMultiArray(shape: shape as [NSNumber], dataType: MLMultiArrayDataType.float32)
        let pointer = multiArray.dataPointer
        let byteCount = multiArray.count * 4
        memset(pointer, Int32(withValue), byteCount)
        return multiArray
    }

}

