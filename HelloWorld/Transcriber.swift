import AVFoundation
import WhisperKit

/// Step-1 test harness: record a short clip to a file, then run Whisper on it.
@MainActor
final class Transcriber: ObservableObject {
    @Published var status = "Idle"
    @Published var transcript = ""
    @Published var isBusy = false
    @Published var isStreaming = false

    private var whisper: WhisperKit?
    private var streamer: AudioStreamTranscriber?
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
            // load: true, else WhisperKit defers loading (incl. the tokenizer streaming needs) to the first transcribe().
            whisper = try await WhisperKit(WhisperKitConfig(model: modelName, load: true))
            status = String(format: "Model ready (%.1fs)", Date().timeIntervalSince(start))
        } catch {
            status = "Model load failed: \(error)"
        }
    }

    func toggleStreaming() async {
        if let streamer {
            await streamer.stopStreamTranscription()
            return
        }
        guard let whisper, let tokenizer = whisper.tokenizer else { status = "Model not loaded"; return }

        let streamer = AudioStreamTranscriber(
            audioEncoder: whisper.audioEncoder,
            featureExtractor: whisper.featureExtractor,
            segmentSeeker: whisper.segmentSeeker,
            textDecoder: whisper.textDecoder,
            tokenizer: tokenizer,
            audioProcessor: whisper.audioProcessor,
            // Strip <|startoftranscript|>/<|0.00|> from text; keep timestamps, streaming uses them to confirm segments.
            decodingOptions: DecodingOptions(skipSpecialTokens: true)
        ) { [weak self] _, state in
            // Confirmed segments are final; the rest is the model's current guess and may still change.
            let confirmed = state.confirmedSegments.map(\.text).joined()
            let pending = state.unconfirmedSegments.map(\.text).joined()
            Task { @MainActor in
                self?.transcript = confirmed + pending
                self?.status = state.isRecording ? "Listening… \(state.currentText)" : "Stopped"
            }
        }
        self.streamer = streamer
        isStreaming = true
        transcript = ""
        do {
            // Returns only after stopStreamTranscription() ends the loop.
            try await streamer.startStreamTranscription()
        } catch {
            status = "Streaming failed: \(error)"
        }
        self.streamer = nil
        isStreaming = false
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
            let wall = Date().timeIntervalSince(start)
            if let tim = results.first?.timings {
                // RTF < 1 means faster than real time; the first run after load is slower (warm-up).
                status = String(
                    format: "Audio %.1fs · encode %.0fms · decode %.0fms · total %.0fms\n"
                        + "RTF %.3f · %.0f tok/s · wall %.2fs",
                    tim.inputAudioSeconds, tim.encoding * 1000, tim.decodingLoop * 1000, tim.fullPipeline * 1000,
                    tim.realTimeFactor, tim.tokensPerSecond, wall
                )
            } else {
                status = String(format: "Done (%.2fs)", wall)
            }
        } catch {
            status = "Transcription failed: \(error)"
        }
    }
}
