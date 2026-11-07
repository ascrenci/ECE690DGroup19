import CoreML
import Foundation
import Tokenizers

/// On-device T5: spoken-math text in, LaTeX out.
/// Models/tokenizer are produced by `model/convert_t5.py` and bundled into the app by the build workflow.
actor T5Translator {
    enum T5Error: LocalizedError {
        case missingResource(String)
        case unexpectedOutput

        var errorDescription: String? {
            switch self {
            case .missingResource(let name): return "Missing bundled resource: \(name) (run model/convert_t5.py)"
            case .unexpectedOutput: return "Unexpected Core ML output"
            }
        }
    }

    private struct Meta: Decodable {
        let prompt: String
        let maxEncoderLength: Int
        let maxDecoderLength: Int
        let padTokenId: Int
        let eosTokenId: Int
        let decoderStartTokenId: Int
        /// T5 can't emit \ ^ { } ~ < ` so the model writes stand-ins; applied in order (model/latex_chars.py).
        let latexDecode: [[String]]
    }

    private let encoder: MLModel
    private let decoder: MLModel
    private let tokenizer: Tokenizer
    private let meta: Meta

    init() async throws {
        guard let folder = Bundle.main.url(forResource: "t5_tokenizer", withExtension: nil) else {
            throw T5Error.missingResource("t5_tokenizer")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        func load(_ name: String) throws -> MLModel {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw T5Error.missingResource(name)
            }
            return try MLModel(contentsOf: url, configuration: config)
        }
        encoder = try load("T5Encoder")
        decoder = try load("T5Decoder")
        meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: folder.appendingPathComponent("t5_meta.json")))
        tokenizer = try await AutoTokenizer.from(modelFolder: folder)
    }

    /// Greedy decoding. The decoder has no KV cache yet, so each step re-runs the whole (padded) prefix.
    func translate(_ text: String, maxNewTokens: Int = 127) throws -> String {
        let encLen = meta.maxEncoderLength
        let decLen = meta.maxDecoderLength

        // Encoder input: prompt + text + </s>, truncated (keeping </s>) and padded.
        var tokens = tokenizer.encode(text: meta.prompt + Self.normalizeSpoken(text), addSpecialTokens: false)
        tokens = Array(tokens.prefix(encLen - 1)) + [meta.eosTokenId]

        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: encLen)], dataType: .int32)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: encLen)], dataType: .int32)
        for index in 0..<encLen {
            inputIds[index] = NSNumber(value: index < tokens.count ? tokens[index] : meta.padTokenId)
            mask[index] = NSNumber(value: index < tokens.count ? 1 : 0)
        }

        let encOut = try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": inputIds,
            "attention_mask": mask
        ]))
        guard let hidden = encOut.featureValue(for: "last_hidden_state")?.multiArrayValue else {
            throw T5Error.unexpectedOutput
        }

        let decoderIds = try MLMultiArray(shape: [1, NSNumber(value: decLen)], dataType: .int32)
        for index in 0..<decLen { decoderIds[index] = 0 }
        let position = try MLMultiArray(shape: [1], dataType: .int32)

        var generated: [Int] = []
        var last = meta.decoderStartTokenId
        for step in 0..<min(maxNewTokens, decLen - 1) {
            decoderIds[step] = NSNumber(value: last)
            position[0] = NSNumber(value: step)
            let out = try decoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [
                "decoder_input_ids": decoderIds,
                "encoder_hidden_states": hidden,
                "encoder_attention_mask": mask,
                "position": position
            ]))
            guard let logits = out.featureValue(for: "logits")?.multiArrayValue else {
                throw T5Error.unexpectedOutput
            }
            let next = Self.argmax(logits)
            if next == meta.eosTokenId { break }
            generated.append(next)
            last = next
        }
        var latex = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
        for step in meta.latexDecode where step.count == 2 {
            latex = latex.replacingOccurrences(of: step[0], with: step[1])
        }
        return latex.trimmingCharacters(in: .whitespaces)
    }

    /// Mirrors latex_chars.normalize_spoken: collapse whitespace, drop trailing . , ; : (case is kept).
    private static func normalizeSpoken(_ text: String) -> String {
        var result = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        while let last = result.last, ".,;: ".contains(last) { result.removeLast() }
        return result
    }

    private static func argmax(_ array: MLMultiArray) -> Int {
        let count = array.count
        var best = 0
        if array.dataType == .float32 {
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            var bestValue = -Float.infinity
            for index in 0..<count where ptr[index] > bestValue {
                bestValue = ptr[index]
                best = index
            }
        } else {
            var bestValue = -Float.infinity
            for index in 0..<count {
                let value = array[index].floatValue
                if value > bestValue {
                    bestValue = value
                    best = index
                }
            }
        }
        return best
    }
}
