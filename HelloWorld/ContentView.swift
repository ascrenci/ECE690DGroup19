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

            Button("Record 5s & Transcribe") {
                Task { await transcriber.recordAndTranscribe() }
            }
            .buttonStyle(.borderedProminent)
            .disabled(transcriber.isBusy)
        }
        .padding()
        .task { await transcriber.loadModel() }
    }
}
