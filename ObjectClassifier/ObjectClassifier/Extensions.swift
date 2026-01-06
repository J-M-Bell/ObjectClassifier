//
//  Extensions.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 1/5/26.
//

import Foundation
import UIKit

extension UIImage {
    static func from(color: UIColor, width: Int, height: Int) -> UIImage? {
        let imageSize = CGSize(width: width, height: height)
        let renderer = UIGraphicsImageRenderer(size: imageSize)
        let image = renderer.image { context in
            color.setFill()
            context.fill(CGRect(origin: .zero, size: imageSize))
        }
        return image
    }
    
    static func from(color: UIColor, size: [Int] ) -> UIImage? {
        let size = CGSize(width: size[0], height: size[1])
        let rect = CGRect(origin: .zero, size: size)
        UIGraphicsBeginImageContextWithOptions(rect.size, false, 0.0)
        guard let context = UIGraphicsGetCurrentContext() else { return nil }
        context.setFillColor(color.cgColor)
        context.fill(rect)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
}

import CoreGraphics

extension CGImage {
    
}



