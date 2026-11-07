@testable import HelloWorld
import XCTest

final class MathViewTests: XCTestCase {
    func testCanRenderValidLatex() {
        XCTAssertTrue(MathView.canRender("\\frac{a}{b}"))
    }

    func testCannotRenderBrokenLatex() {
        XCTAssertFalse(MathView.canRender("\\frac{a"))
    }
}
