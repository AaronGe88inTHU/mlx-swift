//
//  FastTests.swift
//  mlx-swift
//
//  Created by Aaron Ge on 2024/9/3.
//
import Foundation
import MLX
import MLXFFT
import MLXNN
import MLXRandom
import XCTest

@testable import MLXFast

class FastTests: XCTestCase {

    override class func setUp() {
        setDefaultDevice()
    }

    func testFastScaledDotProductAttention() {
        let key = MLXRandom.key(0)
        let R = 20
        let L = 20
        let Dk = 64
        let H = 3

        let scale = (1.0.asMLXArray(dtype: .float32) / sqrt(Dk.asMLXArray(dtype: .float32))).item(
            Float.self)
        let q = truncatedNormal(low: 0, high: 1, [1, H, R, Dk])
        let k = truncatedNormal(low: 0, high: 1, [1, H, L, Dk])
        let v = truncatedNormal(low: 0, high: 1, [1, H, L, Dk])
        let p = matmul((q * scale), k.transposed(0, 1, 3, 2))
        let scores = softmax(p, axis: -1)
        let expected = matmul(scores, v)

        let result = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale)
        assertEqual(result, expected)

    }
}
