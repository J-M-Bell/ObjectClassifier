//
//  ContentView.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/28/25.
//

import SwiftUI
import UniformTypeIdentifiers // Required for UTType

struct ClassifierView: View {
    var classifierVM = ClassifierViewModel()
    @State private var showingFileImporter = false
    @State private var importedFileURLs: [URL] = []
    @State private var importError: Error?
    @State private var importedImage: UIImage?
    @State private var label: String?

    var body: some View {
        VStack(spacing: 16) {
            Text("Welcome to the Object Classifier Mobile App!")

            Button("Import Image") {
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

//            if !importedFileURLs.isEmpty {
//                print("Successfully imported \(importedFileURLs.count) file(s).")
//            }

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

    /// Non-mutating helper that updates @State safely from the main thread
    /// - Parameters: urls: [URL} - array of URL objectss from user
    private func handleImportedFiles(_ urls: [URL]) {
        guard let url = urls.first else { return }

        // Access security-scoped resource
        let needsStop = url.startAccessingSecurityScopedResource()
        defer {
            if needsStop { url.stopAccessingSecurityScopedResource() }
        }

        do { // calls image classification method and updates the View
            let data = try Data(contentsOf: url)
            if let image = UIImage(data: data) {
                DispatchQueue.main.async {
                    self.importedImage = image
                    self.label = self.classifierVM.classifyImage(image: image)
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
}

#Preview {
    ClassifierView()
}
