//
//  ContentView.swift
//  ObjectClassifier
//
//  Created by Justin Bell on 12/26/25.
//

import SwiftUI

struct WelcomeView: View {
    var body: some View {
        NavigationStack {
//            Image(.image)
            Text("Welcome to Object Classifier!")
                .padding(.bottom)
//            Text("The Journaling App of Your Dreams!")
        }
        .padding()
    }
}

#Preview {
    WelcomeView()
}
