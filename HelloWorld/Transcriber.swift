import AVFoundation
import WhisperKit

/// Step-1 test harness: record a short clip to a file, then run Whisper on it.
@MainActor
final class Transcriber: ObservableObject {
    @Published var status = "Idle"
    @Published var transcript = ""
    @Published var isBusy = false

    private var whisper: WhisperKit?
    private var recorder: AVAudioRecorder?
    private let clipURL = FileManager.default.temporaryDirectory.appendingPathComponent("clip.wav")

    // tiny.en / base.en are safe for a free-signed sideloaded app; larger models may exceed memory limits.
    private let modelName = "base.en"

    func loadModel() async {
        guard whisper == nil else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            status = "Loading \(modelName)… (first launch downloads + compiles, can take a minute)"
            let start = Date()
            whisper = try await WhisperKit(WhisperKitConfig(model: modelName))
            status = String(format: "Model ready (%.1fs)", Date().timeIntervalSince(start))
        } catch {
            status = "Model load failed: \(error)"
        }
    }

    func recordAndTranscribe(seconds: Double = 5) async {
        guard whisper != nil else { status = "Model not loaded"; return }
        isBusy = true
        defer { isBusy = false }
        do {
            guard await AVAudioApplication.requestRecordPermission() else {
                status = "Microphone permission denied (enable in Settings)"
                return
            }
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement, options: .defaultToSpeaker)
            try session.setActive(true)

            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16_000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false
            ]
            let recorder = try AVAudioRecorder(url: clipURL, settings: settings)
            self.recorder = recorder
            recorder.record()
            status = "Recording \(Int(seconds))s — speak now…"
            try await Task.sleep(for: .seconds(seconds))
            recorder.stop()
            try session.setActive(false)

            await transcribeClip()
        } catch {
            status = "Recording failed: \(error)"
        }
    }

    private func transcribeClip() async {
        guard let whisper else { return }
        do {
            status = "Transcribing…"
            let start = Date()
            let results = try await whisper.transcribe(audioPath: clipURL.path)
            transcript = results.map(\.text).joined(separator: " ")
            status = String(format: "Done (%.2fs)", Date().timeIntervalSince(start))
        } catch {
            status = "Transcription failed: \(error)"
        }
    }
}
