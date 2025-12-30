//
//  ContentView.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/28/25.
//

import SwiftUI
import UniformTypeIdentifiers // Required for UTType
import CoreML

struct ContentView: View {
    @State private var showingFileImporter = false
    @State private var importedFileURLs: [URL] = []
    @State private var importError: Error?
    @State private var importedImage: UIImage?
    @State private var label: String?

//    let model: ObjectClassifierModel = {
//        do {
//            let config = MLModelConfiguration()
//            return try ObjectClassifierModel(configuration: config)
//        } catch {
//            print(error)
//            fatalError("Error loading model")
//        }
//    }()

    var body: some View {
        VStack(spacing: 16) {
            Text("Welcome to the Object Classifier Mobile App!")

            Button("Import Files") {
                showingFileImporter = true
            }
            .foregroundColor(.white)
            .frame(width: 150, height: 75)
            .background(Color.blue)
            .padding(5)
            .fileImporter(
                isPresented: $showingFileImporter,
                allowedContentTypes: [.image],
                allowsMultipleSelection: false,
                onCompletion: { result in
                    switch result {
                    case .success(let urls):
                        // Update state with the imported URLs
                        self.importedFileURLs = urls
                        // Process files into an image
                        self.handleImportedFiles(urls)
                    case .failure(let error):
                        self.importError = error
                        print(error.localizedDescription)
                    }
                }
            )

            if !importedFileURLs.isEmpty {
                Text("Successfully imported \(importedFileURLs.count) file(s).")
            }

            if let uiImage = importedImage {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 240)
            }
            
            if let label = label {
                Text("Prediction: \(label)")
                    .font(.headline)
            }

            if let error = importError {
                Text("Error: \(error.localizedDescription)")
                    .foregroundStyle(.red)
            }
        }
        .padding()
    }

    enum DataError: Error {
        case invalidInput
        case missingData
        case processingFailed
    }

    
    func createPrediction(image: UIImage) {
        do {
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

            // 1. Create a bitmap context with a specific memory buffer configuration
            guard let context = CGContext(
                data: nil, // Pass nil to let the context allocate the memory
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

            // 2. Draw the image into the context
            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: 32, height: 32))

            // 3. Get the pointer to the pixel data
            guard let data = context.data else {
                print("Could not create data pointer")
                throw DataError.missingData
            }
            let pixelData = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
            

            // 4. Calculate the offset for the specific pixel
            var pixelArray = [Float]()
            var x_r_Array = [Float]()
            var x_g_Array = [Float]()
            var x_b_Array = [Float]()
//            var x_a_Array = [Float]()
            var y_r_Array = [Float]()
            var y_g_Array = [Float]()
            var y_b_Array = [Float]()
//            var y_a_Array = [Float]()
            
            
            for y in 1...32 {
                for x in 1...32 {
                    let pixelInfo: Int = ((bytesPerRow * y) + x * bytesPerPixel)
                    let r = Float(pixelData[pixelInfo]) / 255.0
                    x_r_Array.append(r)
                    let g = Float(pixelData[pixelInfo + 1]) / 255.0
                    x_g_Array.append(g)
                    let b = Float(pixelData[pixelInfo + 2]) / 255.0
                    x_b_Array.append(b)
//                    let a = Float(pixelData[pixelInfo + 3]) / 255.0
//                    x_a_Array.append(b)
                }
                y_r_Array.append(contentsOf: x_r_Array)
                y_g_Array.append(contentsOf: x_g_Array)
                y_b_Array.append(contentsOf: x_b_Array)
//                y_a_Array.append(contentsOf: x_a_Array)
                x_r_Array.removeAll()
                x_g_Array.removeAll()
                x_b_Array.removeAll()
//                x_a_Array.removeAll()
            }
            pixelArray.append(contentsOf: y_r_Array)
            pixelArray.append(contentsOf: y_g_Array)
            pixelArray.append(contentsOf: y_b_Array)
//            pixelArray.append(contentsOf: y_a_Array)
                   
            // The 'shape' must match the total number of elements in the 'floatArray'.
            guard let shapedArray = try? MLShapedArray<Float>(scalars: pixelArray, shape: [1, 32, 32, 3]) else {
                print("Error creating MLShapedArray. Check that the shape matches the array count.")
                throw DataError.missingData
            }

            // 2. Convert the MLShapedArray to an MLMultiArray using its initializer.
            let multiArray = MLMultiArray(shapedArray)
        
            let config = MLModelConfiguration()
            let model = try ObjectClassifierModel(configuration: config)
            // Use the generated convenience prediction API with the model's input name
            // Adjust the input label name `keras_tensor_50` if your model uses a different name
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
            var prediction = ""
            switch indexOfMax {
            case 0:
                prediction = "Airplane"
            case 1:
                prediction = "Automobile"
            case 2:
                prediction = "Bird"
            case 3:
                prediction = "Cat"
            case 4:
                prediction = "Deer"
            case 5:
                prediction = "Dog"
            case 6:
                prediction = "Frog"
            case 7:
                prediction = "Horse"
            case 8:
                prediction = "Ship"
            case 9:
                prediction = "Truck"
            default:
                print("Error occured in switch")
            }2
            DispatchQueue.main.async {
                self.label = prediction
            }
        } catch DataError.missingData {
            print("Error: Missing input data.")
        } catch {
            print(error.localizedDescription)
        }
        
    }
//    func analyzeImage(image: UIImage?) {
//        guard let buffer = image?.resize(size: CGSize(width: 32, height: 32))?.getCVPixelBuffer() else {
//            return
//        }
//        do {
//            let config = MLModelConfiguration()
//            let model = try ObjectClassifierModel(configuration: config)
//            // Use the generated convenience prediction API with the model's input name
//            // Adjust the input label name `keras_tensor_50` if your model uses a different name
//            let input = ObjectClassifierModelInput(keras_tens)
//            let output = try model.prediction(input: input)
//            DispatchQueue.main.async {
//                self.label = output.featureValue(for: "output")?.stringValue
//            }
//        } catch {
//            print(error.localizedDescription)
//        }
//    }
    // Non-mutating helper that updates @State safely from the main thread
    private func handleImportedFiles(_ urls: [URL]) {
        guard let url = urls.first else { return }

        // Access security-scoped resource
        let needsStop = url.startAccessingSecurityScopedResource()
        defer {
            if needsStop { url.stopAccessingSecurityScopedResource() }
        }

        do {
            let data = try Data(contentsOf: url)
            if let image = UIImage(data: data) {
                DispatchQueue.main.async {
                    self.importedImage = image
                    self.createPrediction(image: image)
                }
            } else {
                print("Error: Could not create UIImage from data")
            }
        } catch {
            DispatchQueue.main.async {
                self.importError = error
            }
            print("Error reading file: \(error.localizedDescription)")
        }
    }

    // Async loader that accepts concrete URLs (removed undefined urlString)
//    func loadImage(from urls: [URL]) async -> UIImage? {
//        for url in urls {
//            do {
//                let (data, response) = try await URLSession.shared.data(from: url)
//                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
//                    print("Error: Invalid HTTP response or status code")
//                    continue
//                }
//                if let image = UIImage(data: data) {
//                    return image
//                } else {
//                    print("Error: Could not create UIImage from data")
//                }
//            } catch {
//                print("Error fetching image data: \(error.localizedDescription)")
//            }
//        }
//        return nil
//    }
    
//    import CoreML
//
//    // 1. Define the initial multiarrays
//    let multiArray1 = try MLMultiArray(shape: [1, 5, 7] as [NSNumber], dataType: .int32)
//    let multiArray2 = try MLMultiArray(shape: [2, 5, 7] as [NSNumber], dataType: .int32)
//
//    // ... populate multiArray1 and multiArray2 with data ...
//
//    // 2. Merge them along the first dimension (axis 0)
//    let mergedMultiArray = try MLMultiArray(
//        byConcatenatingMultiArrays: [multiArray1, multiArray2],
//        alongAxis: 0,
//        dataType: .int32
//    )
//
//    // The resulting shape will be [1+2, 5, 7] -> [3, 5, 7]
//    assert(mergedMultiArray.shape == [3, 5, 7] as [NSNumber])

}

#Preview {
    ContentView()
}
