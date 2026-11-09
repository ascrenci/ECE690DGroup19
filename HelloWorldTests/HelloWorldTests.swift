import XCTest
@testable import HelloWorld

final class VocabMappingTests: XCTestCase {
    private let kept = [0, 1, 2, 7, 42]

    func testUnprunedIdsPassThrough() {
        XCTAssertEqual(T5Translator.toModelIds([5, 9], lookup: nil), [5, 9])
        XCTAssertEqual(T5Translator.fromModelIds([5, 9], keptIds: nil), [5, 9])
    }

    func testRoundTripThroughPrunedVocab() {
        let lookup = T5Translator.modelIdLookup(kept)
        let modelIds = T5Translator.toModelIds([7, 42, 1], lookup: lookup)
        XCTAssertEqual(modelIds, [3, 4, 1])
        XCTAssertEqual(T5Translator.fromModelIds(modelIds, keptIds: kept), [7, 42, 1])
    }

    func testIdOutsidePrunedVocabBecomesUnk() {
        XCTAssertEqual(T5Translator.toModelIds([99], lookup: T5Translator.modelIdLookup(kept)), [T5Translator.unkId])
    }
}
