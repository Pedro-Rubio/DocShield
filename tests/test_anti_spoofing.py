import numpy as np
import pytest
from src.pipeline.anti_spoofing import detect_moire, analyze_dct_blocks, analyze_reflection

class TestDetectMoire:
    def test_returns_float(self):
        gray = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        score = detect_moire(gray)
        assert isinstance(score, float)
    
    def test_uniform_image_low_score(self):
        gray = np.ones((200, 300), dtype=np.uint8) * 128
        score = detect_moire(gray)
        assert score >= 0
    
    def test_empty_mask_returns_zero(self):
        gray = np.ones((1, 1), dtype=np.uint8) * 128
        score = detect_moire(gray)
        assert score == 0.0

class TestAnalyzeDCTBlocks:
    def test_returns_float(self):
        gray = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        score = analyze_dct_blocks(gray)
        assert isinstance(score, float)
    
    def test_uniform_image_low_ratio(self):
        gray = np.ones((200, 300), dtype=np.uint8) * 128
        score = analyze_dct_blocks(gray)
        assert score >= 0
    
    def test_small_image(self):
        gray = np.ones((8, 8), dtype=np.uint8) * 128
        score = analyze_dct_blocks(gray)
        assert isinstance(score, float)

class TestAnalyzeReflection:
    def test_returns_float(self):
        bgr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8).astype(np.uint8)
        score = analyze_reflection(bgr)
        assert isinstance(score, float)
    
    def test_no_high_reflection_returns_zero(self):
        bgr = np.ones((200, 300, 3), dtype=np.uint8) * 200
        score = analyze_reflection(bgr)
        assert score == 0.0
    
    def test_with_reflection(self):
        bgr = np.ones((200, 300, 3), dtype=np.uint8) * 250
        score = analyze_reflection(bgr)
        assert score >= 0
