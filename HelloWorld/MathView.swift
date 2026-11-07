import SwiftMath
import SwiftUI

struct MathView: UIViewRepresentable {
    var latex: String
    var fontSize: CGFloat = 24

    static func canRender(_ latex: String) -> Bool {
        var error: NSError?
        return MTMathListBuilder.build(fromString: latex, error: &error) != nil && error == nil
    }

    func makeUIView(context: Context) -> MTMathUILabel {
        let label = MTMathUILabel()
        label.labelMode = .display
        label.textAlignment = .left
        label.displayErrorInline = false
        label.textColor = .label
        label.setContentHuggingPriority(.required, for: .vertical)
        return label
    }

    func updateUIView(_ label: MTMathUILabel, context: Context) {
        label.latex = latex
        label.fontSize = fontSize
        label.invalidateIntrinsicContentSize()
    }

    // shortcut: SwiftMath 1.7.3 can't line-wrap, so long equations scroll sideways; wrap when a release adds it.
    func sizeThatFits(_ proposal: ProposedViewSize, uiView label: MTMathUILabel, context: Context) -> CGSize? {
        label.intrinsicContentSize
    }
}
