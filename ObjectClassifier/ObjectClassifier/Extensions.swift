//
//  Extensions.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 1/5/26.
//

import Foundation
import UIKit

extension UIImage {
    
    /// UIImage extension to create a UIImage of a certain UIColor and dimensions
    /// - Parameters: color: UIColor - color of the new UIImage
    /// - Parameters: width: Int - width of new UIImage
    /// - Parameters: height: Int - height of new UIImage
    ///
    /// - Returns: UIImage of a certain color
    static func from(color: UIColor, width: Int, height: Int) -> UIImage? {
        let imageSize = CGSize(width: width, height: height)
        let renderer = UIGraphicsImageRenderer(size: imageSize)
        let image = renderer.image { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: imageSize))
        }
        return image
    }
}





