//
//  ObjectClassifierModel.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/28/25.
//

import SwiftUI
import Foundation
import CoreML

/// A struct model that holds UI Model class for the ClassifierView
struct ClassifierModel {
    
    /// - TODO: create get and set method for the instance variable
    /// - TODO: create unit test file for get and set methods

    /// list of targets from the model
    let classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    
    /// Object Classifier model object
    let model: ObjectClassifierModel = {
        do {
            let config = MLModelConfiguration()
            return try ObjectClassifierModel(configuration: config)
        } catch {
            print(error)
            fatalError("Error loading model")
        }
    }()
    
    /// image to be inputted to model
    var image: UIImage?
    
    
}
