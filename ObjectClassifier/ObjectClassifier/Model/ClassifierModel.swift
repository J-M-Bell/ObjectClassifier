//
//  ObjectClassifierModel.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/28/25.
//

import SwiftUI
import Foundation
import CoreML

struct ClassifierModel {
    
    let classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    let model: ObjectClassifierModel = {
        do {
            let config = MLModelConfiguration()
            return try ObjectClassifierModel(configuration: config)
        } catch {
            print(error)
            fatalError("Error loading model")
        }
    }()
    var image: UIImage?
    
    
}
