import SwiftUI

struct ContentView: View {
    @StateObject private var transcriber = Transcriber()
    @State private var translator: T5Translator?
    @State private var latex = ""
    @State private var latexStatus = "LaTeX output will appear here"
    @State private var isConverting = false

    var body: some View {
        VStack(spacing: 20) {
            Text("Whisper Test")
                .font(.largeTitle)

            Text(transcriber.status)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            ScrollView {
                Text(transcriber.transcript.isEmpty ? "Transcript will appear here" : transcriber.transcript)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(maxHeight: 300)
            .padding()
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))

            Button(transcriber.isStreaming ? "Stop" : "Start") {
                Task { await transcriber.toggleStreaming() }
            }
            .buttonStyle(.borderedProminent)
            .tint(transcriber.isStreaming ? .red : .accentColor)
            .disabled(transcriber.isBusy)

            // T5 test: convert the current transcript to LaTeX.
            Button("Convert to LaTeX") {
                Task { await convertToLatex() }
            }
            .buttonStyle(.bordered)
            .disabled(transcriber.transcript.isEmpty || isConverting)

            Text(latex.isEmpty ? latexStatus : latex)
                .font(.system(.body, design: .monospaced))
                .foregroundStyle(latex.isEmpty ? .secondary : .primary)
                .textSelection(.enabled)

            // Streaming doesn't expose per-pass timings, so keep the one-shot path for benchmarking.
            Button("Benchmark 5s clip") {
                Task { await transcriber.recordAndTranscribe() }
            }
            .buttonStyle(.bordered)
            .disabled(transcriber.isBusy || transcriber.isStreaming)
        }
        .padding()
        .task { await transcriber.loadModel() }
    }

    private func convertToLatex() async {
        isConverting = true
        defer { isConverting = false }
        do {
            let start = Date()
            if translator == nil {
                latexStatus = "Loading T5…"
                translator = try await T5Translator()
            }
            latexStatus = "Converting…"
            latex = try await translator?.translate(transcriber.transcript) ?? ""
            latexStatus = String(format: "T5 done (%.2fs)", Date().timeIntervalSince(start))
        } catch {
            latex = ""
            latexStatus = "T5 failed: \(error.localizedDescription)"
        }
    }
}
