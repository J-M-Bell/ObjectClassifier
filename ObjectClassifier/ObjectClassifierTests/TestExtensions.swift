//
//  TestExtensions.swift
//  ObjectClassifierTests
//
//  Created by Justin Bell on 1/5/26.
//

import Testing

struct TestExtensions {

//    @Test func <#test function name#>() async throws {
//        // Write your test here and use APIs like `#expect(...)` to check expected conditions.
//    }

}

extension Tag {
    @Tag static var unitTests: Self
    @Tag static var integrationTests: Self
    @Tag static var critical: Self
}
