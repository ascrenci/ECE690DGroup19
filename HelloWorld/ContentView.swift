import SwiftUI

struct ContentView: View {
    @StateObject private var transcriber = Transcriber()

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
}
